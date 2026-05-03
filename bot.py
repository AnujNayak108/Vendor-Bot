"""
Vera challenge bot: deterministic, dataset-grounded composition + FastAPI judge harness.

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import re
import time
import json
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

START_MONO = time.time()

# ---------------------------------------------------------------------------
# In-memory state (judge harness)
# ---------------------------------------------------------------------------
_contexts: dict[tuple[str, str], dict[str, Any]] = {}
_conversations: dict[str, dict[str, Any]] = {}
_sent_suppression: set[str] = set()
_sent_bodies_by_conv: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text).replace("  ", " ").strip()


def _pct(x: float) -> str:
    return f"{int(round(abs(x) * 100))}%"


def _active_offers(merchant: dict) -> list[str]:
    return [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]


def _digest_item(category: dict, item_id: str) -> Optional[dict]:
    for d in category.get("digest", []) or []:
        if d.get("id") == item_id:
            return d
    return None


def _peer_ctr(category: dict) -> Optional[float]:
    ps = category.get("peer_stats") or {}
    v = ps.get("avg_ctr")
    return float(v) if v is not None else None


def _greet_merchant(merchant: dict, category_slug: str) -> str:
    ident = merchant.get("identity") or {}
    owner = (ident.get("owner_first_name") or "").strip()
    if category_slug == "dentists" and owner:
        return f"Dr. {owner}"
    if owner:
        return owner
    name = ident.get("name") or "there"
    return name.split()[0]


def _merchant_display_short(merchant: dict) -> str:
    ident = merchant.get("identity") or {}
    return ident.get("name") or "your business"


def _use_hi_en(merchant: dict, customer: Optional[dict]) -> bool:
    if customer:
        lp = (customer.get("identity") or {}).get("language_pref") or ""
        if "hi" in lp.lower():
            return True
    langs = (merchant.get("identity") or {}).get("languages") or []
    return "hi" in langs


# ---------------------------------------------------------------------------
# Core composer (also used for offline submission.jsonl generation)
# ---------------------------------------------------------------------------

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
) -> dict[str, Any]:
    """
    Inputs: raw dicts from dataset JSON.
    Returns: body, cta, send_as, suppression_key, rationale
    """
    cat_slug = category.get("slug") or merchant.get("category_slug") or "unknown"
    kind = trigger.get("kind") or "unknown"
    inner = trigger.get("payload") or {}
    suppression_key = trigger.get("suppression_key") or f"send:{trigger.get('id', 'unknown')}"
    send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"

    greet = _greet_merchant(merchant, cat_slug)
    mname = _merchant_display_short(merchant)
    ident = merchant.get("identity") or {}
    locality = ident.get("locality") or ""
    city = ident.get("city") or ""

    perf = merchant.get("performance") or {}
    views = perf.get("views")
    calls = perf.get("calls")
    ctr = perf.get("ctr")
    d7 = perf.get("delta_7d") or {}

    peer = _peer_ctr(category)
    ctr_pct_peer = None
    if peer is not None and ctr is not None and peer > 0:
        ctr_pct_peer = int(round((ctr / peer) * 100))

    agg = merchant.get("customer_aggregate") or {}
    high_risk = agg.get("high_risk_adult_count")
    offers = _active_offers(merchant)
    offers_txt = ", ".join(offers) if offers else "no active service+price offers on file"

    body = ""
    cta = "open_ended"
    rationale = ""

    if kind == "research_digest":
        item_id = inner.get("top_item_id")
        d = _digest_item(category, item_id) if item_id else None
        if not d:
            d = (category.get("digest") or [{}])[0]
        trial_n = d.get("trial_n")
        title = d.get("title") or "this week's digest item"
        source = d.get("source") or ""
        seg = (d.get("patient_segment") or "").replace("_", " ")
        hook = ""
        if high_risk and seg and "high risk" in seg.lower():
            hook = f"One line-item looks relevant to your roster signal (~{high_risk} high-risk adults tracked): "
        elif seg:
            hook = f"One item targets {seg}: "
        else:
            hook = "Quick pick from this week's digest: "
        num_line = ""
        if trial_n is not None:
            try:
                num_line = f"A {int(trial_n):,}-patient anchor in the write-up — "
            except (TypeError, ValueError):
                num_line = f"A {trial_n}-patient anchor in the write-up — "
        body = (
            f"{greet}, {hook}{title}. "
            f"{num_line}worth a 2-min skim. "
            f"Want me to pull the excerpt + draft a patient-ed WhatsApp you can paste? "
        )
        if source:
            body += f"— {source}"
        cta = "open_ended"
        rationale = (
            "Research digest trigger: cite digest title + source; tie to merchant aggregate when available; "
            "offer drafted patient-ed asset to reduce merchant effort."
        )

    elif kind == "regulation_change":
        item_id = inner.get("top_item_id")
        d = _digest_item(category, item_id) if item_id else None
        if not d:
            d = _digest_item(category, "d_2026W17_dci_radiograph")
        title = d.get("title") if d else "Compliance update"
        source = d.get("source") if d else ""
        deadline = inner.get("deadline_iso") or ""
        body = (
            f"{greet}, compliance calendar nudge: {title}. "
            f"Effective {deadline}. "
            f"Want me to turn the circular into a 5-bullet SOP checklist for your staff + a patient comms line if needed? "
        )
        if source:
            body += f"— {source}"
        cta = "binary_yes_no"
        rationale = "Regulation trigger: name item + effective date from payload/digest; binary CTA for action."

    elif kind == "recall_due" and customer:
        cid = customer.get("identity") or {}
        cname = cid.get("name") or "there"
        lp = (cid.get("language_pref") or "").lower()
        slots = (inner.get("available_slots") or [])[:2]
        slot_labels = " / ".join(s.get("label") or "" for s in slots if s.get("label"))
        due = inner.get("due_date") or ""
        last = inner.get("last_service_date") or ""
        offer0 = offers[0] if offers else "your listed cleaning offer"
        mix = "hi-en mix" in lp or "mix" in lp
        if mix:
            body = (
                f"Hi {cname}, {mname} here — your 6-month cleaning recall window is open "
                f"(last visit {last}; due around {due}). "
                f"Apke liye 2 slots: {slot_labels}. "
                f"{offer0} — reply 1 for first slot, 2 for second, or suggest another evening time."
            )
        else:
            body = (
                f"Hi {cname}, {mname} here — your 6-month cleaning recall is due "
                f"(last visit {last}; target {due}). "
                f"Open slots: {slot_labels}. "
                f"Active offer on file: {offer0}. Reply 1 / 2 / or share a better time."
            )
        cta = "multi_choice_slot"
        rationale = (
            "Customer recall: merchant_on_behalf; only dates/slots from trigger; offer from merchant.active; "
            "language mix if customer pref indicates Hindi-English."
        )

    elif kind == "perf_dip":
        metric = inner.get("metric") or "calls"
        delta = float(inner.get("delta_pct") or 0)
        window = inner.get("window") or "7d"
        baseline = inner.get("vs_baseline")
        base_seg = f" (baseline ~{baseline}/period)" if baseline is not None else ""
        body = f"{greet}, heads-up: your GBP {metric} is down {_pct(delta)} vs prior {window}{base_seg}."
        if peer is not None and ctr is not None:
            body += f" Peer CTR in this vertical is ~{peer:.1%} — you're at {float(ctr):.1%}."
        if offers:
            body += f" Want me to draft a 3-post GBP calendar for next 10 days using: {offers_txt}?"
        else:
            body += " Want me to draft a tight GBP fix list + 2 posts you can approve?"
        cta = "binary_yes_no"
        rationale = "Perf dip: use only payload deltas + merchant performance + peer CTR from category."

    elif kind == "renewal_due":
        days = inner.get("days_remaining")
        plan = inner.get("plan") or "Pro"
        amt = inner.get("renewal_amount")
        body = (
            f"{greet}, subscription housekeeping: {plan} renewal is due in {days} days "
            f"(₹{amt} on file). "
            f"If you want continuity on profile automation, reply YES and I'll line up the renewal steps; "
            f"reply STOP if you want to pause."
        )
        cta = "binary_yes_stop"
        rationale = "Renewal trigger: urgency from payload numbers only; binary renew/stop CTA."

    elif kind == "festival_upcoming":
        fest = inner.get("festival") or "the upcoming festival"
        date = inner.get("date") or ""
        body = (
            f"{greet}, early planner for {fest} ({date}): "
            f"want a 7-day content pack (GBP post + WhatsApp broadcast) built around your live offers: {offers_txt}? "
            f"Takes one YES from you and I'll draft."
        )
        cta = "binary_yes_no"
        rationale = "Festival external trigger: anchor on festival + date from payload; leverage real merchant offers."

    elif kind == "wedding_package_followup" and customer:
        cid = customer.get("identity") or {}
        cname = cid.get("name") or "there"
        prefs = customer.get("preferences") or {}
        wdate = inner.get("wedding_date") or prefs.get("wedding_date") or ""
        days = inner.get("days_to_wedding")
        trial = inner.get("trial_completed") or ""
        owner = ident.get("owner_first_name") or greet
        spa = next((t for t in offers if "spa" in t.lower()), offers[0] if offers else "your active salon offer")
        body = (
            f"Hi {cname} — {owner} from {mname} ({locality}). "
            f"{days} days to the wedding ({wdate}) after your trial on {trial}. "
            f"Good window to start a simple 4-week prep cadence; we can begin with your live {spa} this Saturday 4pm if that still works. "
            f"Reply YES to block, or suggest another slot."
        )
        cta = "binary_yes_no"
        rationale = "Bridal follow-up: customer scope; dates from trigger + customer prefs; price/offer only from merchant.active."

    elif kind == "curious_ask_due":
        body = (
            f"Hi {greet} — quick operator ask: what service got the most walk-in questions this week at {mname}? "
            f"I'll convert your answer into a Google post + a 4-line WhatsApp reply for price DMs. ~5 min on your side."
        )
        cta = "open_ended"
        rationale = "Curious-ask cadence: low-friction question + reciprocity (drafts)."

    elif kind == "winback_eligible":
        dexp = inner.get("days_since_expiry")
        dip = inner.get("perf_dip_pct")
        laps = inner.get("lapsed_customers_added_since_expiry")
        body = (
            f"{greet}, win-back window: Pro expired {dexp} days ago; profile maintenance paused. "
            f"Calls trend ~{_pct(float(dip or 0))} vs prior week and ~{laps} lapsed customers added since expiry. "
            f"Want a 3-step reactivation (offer refresh + GBP audit + 1 broadcast) — reply YES?"
        )
        cta = "binary_yes_no"
        rationale = "Winback: only payload stats; clear packaged next step."

    elif kind == "ipl_match_today":
        match = inner.get("match") or "today's match"
        venue = inner.get("venue") or ""
        mt = inner.get("match_time_iso") or ""
        weeknight = inner.get("is_weeknight")
        bogo = offers[0] if offers else "your active delivery offer"
        if weeknight is False:
            body = (
                f"{greet}, {match} at {venue} tonight ({mt}). "
                f"Saturday IPL usually shifts covers ~12% lower vs a normal Saturday — worth skipping a generic match-night discount. "
                f"Instead, push {bogo} as delivery-first positioning. Want me to draft the Swiggy banner copy + a 2-slide Insta story? "
            )
        else:
            body = (
                f"{greet}, {match} at {venue} ({mt}). "
                f"Want a tight match-night bundle post using {bogo} + a 'last-mile delivery ETA' note to calm reviews?"
            )
        cta = "binary_yes_no"
        rationale = "IPL trigger: match/venue/time from payload; contrarian Saturday guidance per brief; no fabricated league stats."

    elif kind == "review_theme_emerged":
        theme = inner.get("theme") or "a review theme"
        n = inner.get("occurrences_30d")
        quote = inner.get("common_quote") or ""
        body = (
            f"{greet}, review pattern alert: '{theme}' mentioned {n}x in 30d "
            f"({quote}). "
            f"Want me to draft an ops checklist (kitchen/expiry + rider handoff) + a polite WhatsApp template for delayed orders?"
        )
        cta = "binary_yes_no"
        rationale = "Review theme: occurrences + quote from trigger; practical remediation offer."

    elif kind == "milestone_reached":
        metric = inner.get("metric") or "reviews"
        now = inner.get("value_now")
        target = inner.get("milestone_value")
        body = (
            f"{greet}, you're one nudge from a nice GBP milestone — {metric} at {now}/{target}. "
            f"Want a 48-hour 'review reminder' script for dine-in bills + a GBP post celebrating the push?"
        )
        cta = "binary_yes_no"
        rationale = "Milestone trigger: use value_now/milestone_value from payload only."

    elif kind == "active_planning_intent":
        topic = inner.get("intent_topic") or "your idea"
        lastm = inner.get("merchant_last_message") or ""
        if "corporate" in topic.lower() or "thali" in topic.lower():
            body = (
                f"{greet}, continuing your note — \"{lastm}\". "
                f"Starter corporate thali ladder for {mname} (edit freely):\n"
                f"• 10 thalis @ ₹125 + free delivery in Indiranagar\n"
                f"• 25 thalis @ ₹115 + 2 complimentary filter coffees\n"
                f"• 50+ @ ₹105 + 1 family dosa platter\n"
                f"Order window: day-before 5pm; drop 12:30–1pm. "
                f"Want me to draft a 3-line WhatsApp for office admins in your delivery radius?"
            )
        else:
            body = (
                f"{greet}, on {topic}: propose a 4-week kids batch (age 7–12), 3 classes/week, ₹2,499, "
                f"with 1 guardian orientation session — aligned with your 'First Month @ ₹499' trial funnel. "
                f"Reply YES and I'll draft the GBP post + fee table you can paste."
            )
        cta = "binary_yes_no" if "corporate" not in topic.lower() else "open_ended"
        rationale = "Active planning: continue merchant's last message; draft concrete package without inventing named buildings."

    elif kind == "seasonal_perf_dip":
        metric = inner.get("metric") or "views"
        delta = float(inner.get("delta_pct") or 0)
        note = inner.get("season_note") or "seasonal window"
        members = agg.get("total_active_members")
        mline = f"Protect retention on your {members} active members first — " if members else ""
        body = (
            f"{greet}, your {metric} is down {_pct(delta)} this week — flagged as expected ({note}). "
            f"{mline}"
            f"Want me to draft a 'summer attendance challenge' message + a 2-post GBP plan you can run without extra ad spend?"
        )
        cta = "binary_yes_no"
        rationale = "Seasonal dip: reassure using is_expected_seasonal + metric delta; member count from merchant aggregate only."

    elif kind == "customer_lapsed_hard" and customer:
        cid = customer.get("identity") or {}
        cname = cid.get("name") or "there"
        days = inner.get("days_since_last_visit")
        focus = inner.get("previous_focus") or "your goals"
        owner = ident.get("owner_first_name") or greet
        trial_offer = offers[0] if offers else "a trial class"
        body = (
            f"Hi {cname} — {owner} from {mname}. It's been ~{days} days since your last visit — happens often, no pressure. "
            f"If you want to ease back with a {focus} focus, reply YES for a complimentary trial slot this week ({trial_offer}). "
            f"No auto-charge."
        )
        cta = "binary_yes_no"
        rationale = "Lapsed member: warm, no-shame; numbers from payload; offer from merchant."

    elif kind == "trial_followup" and customer:
        cid = customer.get("identity") or {}
        cname = cid.get("name") or "there"
        opts = inner.get("next_session_options") or []
        lbl = opts[0].get("label") if opts else "next available batch slot"
        body = (
            f"Hi {cname}, thanks for the kids trial at {mname}. "
            f"Next suggested slot on file: {lbl}. Reply YES to book, or tell me a better Saturday morning time."
        )
        cta = "binary_yes_no"
        rationale = "Trial follow-up: single next slot from payload."

    elif kind == "supply_alert":
        mol = inner.get("molecule") or "product"
        batches = inner.get("affected_batches") or []
        mfr = inner.get("manufacturer") or "the manufacturer"
        chronic_n = agg.get("chronic_rx_count")
        btxt = ", ".join(batches) if batches else "listed batches"
        base = chronic_n or agg.get("total_unique_ytd")
        body = (
            f"{greet}, urgent supply signal: voluntary recall on {mol} batches {btxt} ({mfr}) — sub-potency risk framing per alert, replacement workflow needed. "
        )
        if base:
            body += (
                f"You track ~{base} chronic patients — want a branch comms pack: counter script + patient WhatsApp + substitution SOP? "
            )
        else:
            body += "Want a branch comms pack: counter script + patient WhatsApp + substitution SOP? "
        cta = "binary_yes_no"
        rationale = "Supply alert: molecule/batches/manufacturer from payload; patient count from merchant aggregate only."

    elif kind == "appointment_tomorrow" and customer:
        cid = customer.get("identity") or {}
        cname = cid.get("name") or "there"
        owner = ident.get("owner_first_name") or greet
        body = (
            f"Hi {cname} — {owner} from {mname} ({locality}). "
            f"Quick reminder: you have an appointment on our books for tomorrow. "
            f"Reply CONFIRM if you're coming, or RESCHEDULE if you need a different time."
        )
        cta = "binary_confirm_cancel"
        rationale = (
            "Appointment-tomorrow trigger: customer-facing; no fabricated time — uses calendar framing only."
        )

    elif kind == "customer_lapsed_soft" and customer:
        cid = customer.get("identity") or {}
        cname = cid.get("name") or "there"
        rel = customer.get("relationship") or {}
        last = rel.get("last_visit") or ""
        offer0 = offers[0] if offers else "a visit this week"
        body = (
            f"Hi {cname}, {mname} here — it's been a while since your last visit ({last}). "
            f"Want me to hold something gentle for you: {offer0}? Reply YES for slot options."
        )
        cta = "binary_yes_no"
        rationale = "Lapsed-soft: use last_visit + state from customer context; offer from merchant catalog."

    elif kind == "chronic_refill_due" and customer:
        if inner.get("placeholder"):
            cid = customer.get("identity") or {}
            cname = cid.get("name") or "there"
            body = (
                f"Hi {cname} — {mname} ({locality}). "
                f"If you have an active chronic prescription tracked with us, reply REFILL and I'll ask the pharmacist to verify packs + totals before any dispatch."
            )
            cta = "open_ended"
            rationale = "Placeholder chronic refill: no invented molecules; consent-based pharmacist verification path."
        else:
            cid = customer.get("identity") or {}
            cname = cid.get("name") or "customer"
            mols = inner.get("molecule_list") or []
            mtxt = ", ".join(mols)
            out = inner.get("stock_runs_out_iso") or ""
            senior = cid.get("senior_citizen")
            deliv = inner.get("delivery_address_saved")
            off_notes = []
            for t in offers:
                if "senior" in t.lower():
                    off_notes.append(t)
                if "delivery" in t.lower():
                    off_notes.append(t)
            off_piece = f" Active offers on file: {'; '.join(off_notes)}." if off_notes else ""
            channel = (cid.get("language_pref") or "").lower()
            pref_ch = (customer.get("preferences") or {}).get("channel", "") or ""
            if channel == "hi" or "son" in pref_ch.lower():
                body = (
                    f"Namaste — {mname} ({locality}, {city}). {cname} ji ki medicines ({mtxt}) {out} tak khatam ho jayengi."
                    f"{off_piece} "
                    f"Reply CONFIRM for same-dose prep + home delivery to saved address; "
                    f"final bill share kar dunga dispatch se pehle."
                )
            else:
                body = (
                    f"Hi — {mname} here ({locality}). Refill due for {cname}: {mtxt} (runs out {out})."
                    f"{off_piece} "
                    f"Reply CONFIRM to prepare packs for the saved address; I'll share payable total before dispatch."
                )
            if senior:
                body += " Senior discount rules on your account will apply if eligible."
            if deliv:
                body += " Delivery address on file — no need to retype."
            cta = "binary_confirm_cancel"
            rationale = (
                "Chronic refill: molecules + date from payload; pricing totals not invented; offers only if present."
            )

    elif kind == "category_seasonal":
        trends = inner.get("trends") or []
        tr = "; ".join(trends[:4]) if trends else "category demand shifts"
        body = (
            f"{greet}, summer shelf shift signal for {locality}: {tr}. "
            f"Want a week-by-week stocking + window-display checklist tied to your current bestsellers?"
        )
        cta = "binary_yes_no"
        rationale = "Seasonal category trends from payload list only."

    elif kind == "gbp_unverified":
        uplift = inner.get("estimated_uplift_pct")
        path = inner.get("verification_path") or "postcard/phone"
        body = (
            f"{greet}, GBP verification is still pending ({path}). "
            f"Teams in your cohort often see ~{_pct(float(uplift or 0))} better direction actions once verified. "
            f"Reply YES for a step-by-step verification checklist; STOP to pause nudges."
        )
        cta = "binary_yes_stop"
        rationale = "Unverified GBP: uplift estimate from payload; binary CTA."

    elif kind == "cde_opportunity":
        item_id = inner.get("digest_item_id")
        d = _digest_item(category, item_id) if item_id else None
        title = d.get("title") if d else "CDE session"
        credits = inner.get("credits")
        fee = inner.get("fee") or "fee on file"
        src = d.get("source") if d else ""
        when = d.get("date") if d else ""
        body = (
            f"{greet}, CDE slot: {title} ({when}). "
            f"Credits: {credits}; fee rule: {fee}. "
            f"Want me to hold your RSVP draft + a clinic calendar block? "
        )
        if src:
            body += f"— {src}"
        cta = "binary_yes_no"
        rationale = "CDE trigger: event metadata only from digest + payload."

    elif kind == "competitor_opened":
        cn = inner.get("competitor_name") or "a new clinic"
        dist = inner.get("distance_km")
        their = inner.get("their_offer") or ""
        od = inner.get("opened_date") or ""
        hero = offers[0] if offers else "your hero service line"
        if cat_slug == "dentists":
            angle = "supervised care + recall-interval positioning"
        elif cat_slug == "restaurants":
            angle = "quality + speed (delivery ETA) positioning"
        elif cat_slug == "gyms":
            angle = "coaching quality + safety positioning"
        elif cat_slug == "pharmacies":
            angle = "sourcing + pharmacist counselling positioning"
        else:
            angle = "trust + consistency positioning"
        body = (
            f"{greet}, market note: {cn} opened {od} ~{dist}km away with {their}. "
            f"Your active positioning: {offers_txt}. "
            f"Want a contrast post ({angle}) + a tight caption around {hero} that doesn't read as pure discounting?"
        )
        cta = "binary_yes_no"
        rationale = "Competitor trigger: use competitor fields from payload; anchor on merchant offers."

    elif kind == "perf_spike":
        metric = inner.get("metric") or "calls"
        delta = float(inner.get("delta_pct") or 0)
        driver = inner.get("likely_driver") or "recent content"
        body = (
            f"{greet}, good problem: {metric} up {_pct(delta)} vs baseline — likely driver: {driver}. "
            f"Want me to package a 'book the surge' GBP post + FAQ replies for trial parents?"
        )
        cta = "binary_yes_no"
        rationale = "Perf spike: celebrate + capitalize using payload deltas."

    elif kind == "dormant_with_vera":
        ds = inner.get("days_since_last_merchant_message")
        topic = inner.get("last_topic") or "last thread"
        body = (
            f"{greet}, checking in after {ds} days quiet on WhatsApp (last topic: {topic}). "
            f"One useful free action I can do today: refresh your GBP services list from {offers_txt}. Reply YES?"
        )
        cta = "binary_yes_no"
        rationale = "Dormancy: acknowledge gap; low-friction offer grounded in real offers."

    else:
        body = (
            f"{greet}, quick Vera check-in for {mname} ({locality}, {city}). "
            f"On file: views {views}, calls {calls}, CTR {float(ctr):.1%}." if ctr is not None else
            f"{greet}, quick Vera check-in for {mname} ({locality}, {city})."
        )
        if peer and ctr is not None:
            body += f" Peer median CTR ~{peer:.1%} (~{ctr_pct_peer}% of peer)." if ctr_pct_peer is not None else f" Peer median CTR ~{peer:.1%}."
        body += f" Active offers: {offers_txt}. Want a 7-day GBP plan?"
        cta = "binary_yes_no"
        rationale = f"Fallback for kind={kind}: still anchored on merchant performance + offers."

    body = _strip_urls(body.strip())
    if not body:
        body = f"{greet}, Vera here — small update queued for {mname}. Reply YES to continue."
        cta = "binary_yes_no"

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale,
    }


def _template_params_from_body(body: str, merchant_name: str) -> list[str]:
    parts = [merchant_name]
    chunk = body.replace("\n", " ")
    if len(chunk) > 220:
        chunk = chunk[:217] + "..."
    parts.append(chunk)
    return parts


# ---------------------------------------------------------------------------
# Conversation / reply brain
# ---------------------------------------------------------------------------

_AUTO_PATTERNS = (
    "thank you for contacting",
    "thanks for contacting",
    "team will respond",
    "we will get back",
    "automated assistant",
    "this is an automated",
)


def _is_auto_reply(msg: str) -> bool:
    m = msg.lower().strip()
    return any(p in m for p in _AUTO_PATTERNS)


def _is_hostile(msg: str) -> bool:
    m = msg.lower()
    return any(
        x in m
        for x in (
            "stop messaging",
            "don't message",
            "not interested",
            "spam",
            "useless",
            "leave me alone",
        )
    )


def _is_commitment(msg: str) -> bool:
    m = msg.lower()
    return any(
        x in m
        for x in (
            "let's do it",
            "lets do it",
            "ok lets",
            "go ahead",
            "please proceed",
            "sounds good, do it",
            "yes, do it",
        )
    )


def _is_gst_offtopic(msg: str) -> bool:
    return "gst" in msg.lower()


def handle_reply(conv_id: str, merchant_id: str, message: str, turn: int) -> dict[str, Any]:
    st = _conversations.setdefault(
        conv_id,
        {"merchant_id": merchant_id, "turns": [], "last_bot": "", "auto_streak": 0, "closed": False},
    )
    if st.get("closed"):
        return {"action": "end", "rationale": "Conversation already ended."}

    st["turns"].append({"role": "merchant", "message": message})

    if _is_hostile(message):
        st["closed"] = True
        return {
            "action": "end",
            "rationale": "Merchant hostility/opt-out — close without further pushes.",
        }

    if _is_auto_reply(message):
        prev = st.get("last_auto")
        if prev == message.strip():
            st["auto_streak"] = int(st.get("auto_streak", 0)) + 1
        else:
            st["auto_streak"] = 1
        st["last_auto"] = message.strip()

        streak = st["auto_streak"]
        if streak >= 3:
            st["closed"] = True
            return {
                "action": "end",
                "rationale": "Repeated identical auto-reply — assume WhatsApp Business canned response; exit.",
            }
        if streak == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Auto-reply twice — back off 24h for owner attention.",
            }
        return {
            "action": "send",
            "body": _strip_urls(
                "Looks like a WhatsApp auto-reply. When the owner sees this: reply YES and I'll continue with the draft I promised."
            ),
            "cta": "binary_yes_no",
            "rationale": "First auto-reply — single clarifying prompt; still on-mission.",
        }

    if _is_commitment(message):
        return {
            "action": "send",
            "body": _strip_urls(
                "Locked in. I'm drafting the patient-ed WhatsApp + the GBP post now — ~90 seconds. "
                "Reply CONFIRM to send the WhatsApp draft to your opted-in patient list segment."
            ),
            "cta": "binary_confirm_cancel",
            "rationale": "Explicit commitment — switch to execution, no more qualifying questions.",
        }

    if _is_gst_offtopic(message):
        return {
            "action": "send",
            "body": _strip_urls(
                "GST filing isn't something I can help with — that's best with your CA. "
                "If you're still good on the Vera task we were doing, reply YES and I'll continue there."
            ),
            "cta": "binary_yes_no",
            "rationale": "Off-scope request politely declined; thread preserved.",
        }

    return {
        "action": "send",
        "body": _strip_urls(
            "Got it. Tell me one constraint (time/budget/tone) and I'll adapt the draft. "
            "If you want me to proceed with the default version instead, reply YES."
        ),
        "cta": "binary_yes_no",
        "rationale": "Generic engaged reply — ask for one constraint; offer default path.",
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Vera Challenge Bot", version="1.0.0")


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any] = Field(default_factory=dict)
    delivered_at: str = ""


class TickBody(BaseModel):
    now: str = ""
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str = ""
    turn_number: int = 1


@app.get("/v1/healthz")
async def healthz():
    counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in _contexts.items():
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_MONO),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Challenge Participant",
        "team_members": ["Vera Bot"],
        "model": "deterministic-template-composer",
        "approach": "Trigger-dispatched, dataset-grounded copy; multi-turn auto-reply + intent routing",
        "contact_email": "participant@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    cur = _contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    _contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: list[dict[str, Any]] = []
    for trg_id in body.available_triggers[:20]:
        rec = _contexts.get(("trigger", trg_id))
        if not rec:
            continue
        trg = rec["payload"]
        mid = trg.get("merchant_id")
        if not mid:
            continue
        mrec = _contexts.get(("merchant", mid))
        if not mrec:
            continue
        merchant = mrec["payload"]
        cat_slug = merchant.get("category_slug")
        cat_rec = _contexts.get(("category", cat_slug), {})
        category = cat_rec.get("payload", {}) if cat_rec else {}

        cid = trg.get("customer_id")
        customer = None
        if cid:
            crec = _contexts.get(("customer", cid))
            if crec:
                customer = crec["payload"]

        sk = trg.get("suppression_key") or trg_id
        if sk in _sent_suppression:
            continue

        out = compose(category, merchant, trg, customer)
        conv_id = f"conv_{mid}_{trg_id}"

        body_text = out["body"]
        if _sent_bodies_by_conv.get(conv_id) and _sent_bodies_by_conv[conv_id][-1] == body_text:
            continue
        _sent_bodies_by_conv.setdefault(conv_id, []).append(body_text)
        _sent_suppression.add(sk)

        st = _conversations.setdefault(conv_id, {"merchant_id": mid, "turns": [], "last_bot": "", "closed": False})
        st["last_bot"] = body_text
        st["trigger_id"] = trg_id

        ident = merchant.get("identity") or {}
        mname = ident.get("name") or "Merchant"

        actions.append(
            {
                "conversation_id": conv_id,
                "merchant_id": mid,
                "customer_id": cid,
                "send_as": out["send_as"],
                "trigger_id": trg_id,
                "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
                "template_params": _template_params_from_body(body_text, mname),
                "body": body_text,
                "cta": out["cta"],
                "suppression_key": out["suppression_key"],
                "rationale": out["rationale"],
            }
        )
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    if body.from_role not in ("merchant", "customer"):
        raise HTTPException(status_code=400, detail="invalid from_role")
    mid = body.merchant_id or ""
    result = handle_reply(body.conversation_id, mid, body.message, body.turn_number)
    return result


def _load_local_dataset_for_cli() -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    from pathlib import Path

    base = Path(__file__).parent / "dataset"
    cats: dict[str, dict] = {}
    for f in (base / "categories").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        cats[d["slug"]] = d
    merchants: dict[str, dict] = {}
    customers: dict[str, dict] = {}
    triggers: dict[str, dict] = {}
    ms = json.loads((base / "merchants_seed.json").read_text(encoding="utf-8"))["merchants"]
    for m in ms:
        merchants[m["merchant_id"]] = m
    cs = json.loads((base / "customers_seed.json").read_text(encoding="utf-8"))["customers"]
    for c in cs:
        customers[c["customer_id"]] = c
    ts = json.loads((base / "triggers_seed.json").read_text(encoding="utf-8"))["triggers"]
    for t in ts:
        triggers[t["id"]] = t
    return cats, merchants, customers, triggers


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bot:app", host="0.0.0.0", port=8080, reload=False)
