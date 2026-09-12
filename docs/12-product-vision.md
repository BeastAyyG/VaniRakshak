# 12 — Product vision (internal)

**Internal. Not for the README or any public surface.** Moved out of the
top-level README on 2026-09-13 because it describes the go-to-market plan in
enough detail for someone else to build it. The repository is private; keep it
that way before sharing this file.

The honesty notes below date from before the countermeasure was trained. For
current measured numbers see `docs/08-cm-training-plan.md` §12.7 and
`docs/10-quality-and-hardening-plan.md`.

---

## Purpose & Vision

VaniRakshak is, at its core, a **voice-authenticity system**: it decides whether the voice
on a call is the *enrolled, real person* it claims to be, or someone (or something) else.
Everything below — the streaming pipeline, the risk tiers, the transaction interlock — exists
to answer that one question reliably and fast.

### Where it started

The system was first built for a single, high-stakes purpose: **guarantee that a call from
Narendra Modi (Modi ji) cannot be faked or impersonated.** One enrolled voice profile — eight
real recordings of Modi ji as captured in `data/anchor_asv_embedding.json`
(`user_id: narendra_modi`) — becomes the anchor. Any inbound or live call claiming to be him is
checked against that anchor. If the voice does not match, it is flagged as *not* the enrolled
Modi voice. The value proposition is deliberately simple and strong: **no one can stand up a
cloned or impersonated "Modi ji" call and have it pass.**

### How it scales: a telecom family-voice safety feature

The same mechanism generalises far beyond one public figure. Carriers already experiment with
scam-detecting calls (for example Airtel's fraud-warning prompts on suspected spam). VaniRakshak
turns that idea inside-out into a **family safety feature sold on a family SIM plan**:

1. **Enroll the family.** When a household takes a family SIM plan, each member records a few
   real voice samples into the carrier's secure enrollment store. These are genuine,
   consenting, in-the-clear recordings — the *ground truth* for that person.
2. **Verify every inbound call.** When a call arrives claiming to be "Dad" or "Mom," the system
   scores the caller's voice against the enrolled sample for that family member.
3. **Real family member, or deepfake?** A strong match ⇒ the caller is the real enrolled family
   member. A weak match ⇒ either a different person *or* a synthetic clone pretending to be them.
   The carrier surfaces a calm verdict ("VERIFIED: this is really Dad" / "COULD NOT VERIFY:
   possible impersonation") and, where configured, steps up to a challenge or blocks the
   transaction.

This is the product carriers can ship: **"know for sure the voice on the line is your real
family member, not a deepfake."** It directly targets the fastest-growing phone-fraud vector —
AI voice cloning used to scam relatives — and gives the carrier a differentiated family-safety
upsell.

> **Honesty note — current model vs. the vision.** The shipped checkpoint proves the
> *mechanism*, not the full claim. It is a one-identity verifier (Modi ji) trained on **zero**
> cloned or synthetic audio, so:
> - It correctly answers *"is this the enrolled voice?"* for the enrolled identity.
> - It has **not** been validated to distinguish a *real non-enrolled human* from a
>   *high-quality deepfake* of that same person, because no synthetic audio was in training.
>   A "not verified" verdict today means "this is not the enrolled voice" — which could be a
>   stranger, a family member not yet enrolled, *or* a clone.
> - Generalising to the family-SIM scenario requires **enrolling each family member on real
>   samples** and **adding cloned/synthetic data to training + evaluation** so the model is
>   actually tested against the thing it must catch.
>
> Treat the family-safety framing as the *design target and deployment model*, validated
> end-to-end only for the single enrolled identity so far. See
> [`ml/README.md`](ml/README.md) for the measured figures.

