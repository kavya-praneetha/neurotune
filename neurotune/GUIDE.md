# neurotune — A Plain-English Guide

A companion to `README.md`. That file is written for someone about to change the
code. This one is for someone who wants to understand what the system does and
why it's built this way — no EEG or machine-learning background assumed.

---

## 1. The one-sentence version

**Read someone's brainwaves, work out how stressed they are, play them music
chosen to calm *them specifically*, measure whether it worked, and use that
measurement to choose better next time.**

That last part is the whole point. Plenty of systems detect stress. Plenty
recommend music. This one closes the loop between them.

---

## 2. The big idea: a loop, not a pipeline

```
        ┌───────────────────────────────────────────────────┐
        │                                                   │
        ▼                                                   │
   ┌─────────┐    ┌──────────┐    ┌────────┐    ┌────────┐  │
   │  BRAIN  │───►│  DETECT  │───►│ CHOOSE │───►│  PLAY  │  │
   │ signals │    │  stress  │    │ music  │    │   it   │  │
   └─────────┘    └──────────┘    └────────┘    └────────┘  │
                                                     │      │
                                                     ▼      │
                                               ┌──────────┐ │
                                               │ MEASURE  │─┘
                                               │the change│  learn
                                               └──────────┘
```

A straight pipeline would stop at "play it." The arrow going back is what makes
this a *closed loop*: every session teaches the system something about this
particular person, and the next session is better for it.

---

## 3. The study this is built around

Twenty people. Each comes in four separate times. Each visit is 15 minutes,
split into four blocks:

| Block | Length | What happens |
|---|---|---|
| **Baseline** | 3 min | Sit quietly. This is what "calm you" looks like. |
| **Stress** | 4 min | A stressful task. Now we know what "stressed you" looks like. |
| **Music** | 5 min | An Indian classical raga plays. |
| **Post-music** | 3 min | Sit quietly again. Did it help? |

Throughout, a cap with **32 electrodes** records electrical activity from the
scalp, 512 times per second. Three electrodes matter most — **Fp1, Fz, Fp2**,
across the forehead — because that's where stress shows up most clearly.

Twenty people × four visits = **80 sessions**. Chopped into 2-second slices,
that's **36,000 slices** of brain activity to work with.

---

## 4. What we look for in the signal

Brainwaves are described by how fast they wiggle. Three speeds matter:

| Wave | Speed | Rough meaning | Under stress |
|---|---|---|---|
| **Theta** | 4–8 Hz | mental effort, concentration | rises a little |
| **Alpha** | 8–13 Hz | calm, relaxed wakefulness | **falls** |
| **Beta** | 13–30 Hz | alertness, arousal, tension | **rises** |

So the signature of stress is: *alpha down, beta up*. Which is why the
**beta-to-alpha ratio** works as a single "how wound up is this person" number —
it captures both halves at once and is steadier than either alone.

When music works, we expect this to run backwards: alpha climbing, beta/alpha
falling.

---

## 5. The seven stages, in plain words

### Stage 1 — Clean up the recording

**Problem:** raw EEG is filthy. Every eye blink produces a spike far bigger than
any brainwave. Jaw clenches add noise. Mains electricity hums underneath.

**What we do:**

1. **Filter** — discard anything below 1 Hz or above 45 Hz. Nothing useful lives
   outside that range; plenty of junk does.
2. **ICA** — the clever part. Imagine recording a room where several people talk
   at once, using several microphones. Each mic hears everyone, in different
   proportions. ICA works backwards from those mixtures to recover the
   individual voices. Here the "voices" are alpha rhythm, beta rhythm, eye
   blinks, muscle tension — and once separated we simply delete the blink voice
   and reassemble the rest.
3. **Slice** into 2-second chunks.
4. **Normalise per person** — skull thickness varies, so raw signal size varies
   hugely between people. We rescale each person's data relative to themselves.

**One subtle thing worth understanding.** Step 4 is necessary for the AI (it
shouldn't waste capacity learning that person #7 has a thick skull), but it
*destroys the real-world units*. After rescaling you can no longer say "alpha
rose by 1.3 microvolts-squared," because microvolts are gone.

So the system deliberately keeps **two copies** of every slice:

- one in **real units** — used for every statistic and every claim about effect size
- one **rescaled** — used only to feed the AI

Mixing these up is an easy, invisible mistake that would silently corrupt every
reported result. Keeping them physically separate makes it impossible.

---

### Stage 2 — Turn brainwaves into pictures

Each 2-second slice becomes a **spectrogram** — think of it as sheet music for
brainwaves. Time runs left to right, frequency bottom to top, brightness shows
how much of that frequency was present at that moment.

Each slice becomes a small image: **3 channels × 45 frequencies × 50 time
steps**. Three "channels" because we use three electrodes — the same way a
colour photo has red, green and blue channels.

Why bother? Once brain activity is a picture, we can use image-recognition AI on
it, and those techniques are very good.

Alongside this we *also* compute the old-fashioned numbers — plain alpha, beta
and theta levels and their ratios. Two reasons: they're interpretable (a
clinician can read them), and they give an honest baseline to compare the AI
against.

---

### Stage 3 — Teach a model to recognise stress

Two models, same job, so we can see which approach is better.

**Model A: CNN-LSTM** (770,692 adjustable parameters)

Two parts in sequence:

- The **CNN** looks at one 2-second spectrogram and answers *"what does this
  moment look like?"* Same family of technique that recognises faces in photos.
- The **LSTM** reads the CNN's answers in order and tracks *"how is this person's
  state changing?"* It has a memory, so it can notice someone getting steadily
  more tense.

The CNN sees snapshots; the LSTM watches the story unfold.

**Model B: Transformer** (507,652 parameters)

Same CNN front end, different back end. Instead of reading in order it uses
**attention** — looking at all 20 slices at once and deciding which matter. Its
advantage is that we can *see* what it decided. Ask "which part made you say
stressed?" and it will show you. That matters for trust: a clinician is far more
likely to accept a system that can point at its own reasoning.

**Both answer two questions at once** ("multi-task"):

1. Which block is this — baseline, stress, or post-music? *(a category)*
2. How stressed, 0 to 10? *(a number)*

Doing both beats either alone. Learning to output a number forces the model to
understand *degree*, not just *category*, which sharpens the category judgement.

---

### Stage 4 — Work out what *kind* of music helps

Detecting stress is only useful if we know what to do about it.

We take ~120 raga tracks and describe each two ways:

- **What a musician would say:** tempo (beats per minute), how busy the
  percussion is, whether there's a *drone* (the continuous sustained note under
  Indian classical music), which scale the raga uses.
- **What a computer measures:** brightness of the sound, its texture fingerprint
  (MFCCs), how "buzzy" it is.

Then we ask statistically: *when someone heard a track with these properties,
how much did their alpha change?*

The answer is a set of coefficients — e.g. "each 10 BPM slower is worth about
+0.5 units of alpha." That's a testable, quotable statement about mechanism,
not just a black-box prediction.

---

### Stage 5 — Choose the music (two stages)

**Stage 1: the filter.** Fast and simple, like the sidebar on a shopping site.
Stress above 7 out of 10 → only show tracks under 80 BPM, calm percussion, with
a drone. Narrows 120 tracks to ~25 candidates. It guarantees *musical
appropriateness* and nothing more — it knows nothing about who you are.

**Stage 2: the personal ranking.** Now reorder those 25 for *this specific
person*, using what we learned from their earlier visits. The system predicts,
for each candidate, how much this person's alpha will rise — highest first.

**Why two stages:** filtering is cheap and can run over the whole library;
ranking is expensive and only needs to run on the shortlist.

**One counter-intuitive but essential detail: the system must sometimes choose
randomly.**

If it always plays its single best guess, everyone hears the same track, and it
never discovers that *you* happen to respond to a faster tempo than most people.
It would never learn anything personal. So roughly 30% of the time it
deliberately tries something else. This is called *exploration*, and without it
the personalisation feature is dead on arrival — it measures as zero improvement
no matter how good the underlying model is.

(We found this the hard way. See §8.)

---

### Stage 6 — Explain the recommendation

A recommendation nobody trusts doesn't get used. So when the system suggests a
track, it also produces a written justification grounded in published research.

We keep a library of short passages from scientific literature and clinical
guidance. When a recommendation is made, the system searches that library for
the most relevant passages and quotes them.

The search is *meaning-based*, not keyword-based. Ask for "slow calming music"
and it will find a passage about "reduced tempo lowering arousal" even though
they share no words.

**Important safety choice:** the explanation is *assembled from quotes*, never
freely written by an AI. A language model asked to write a clinical
justification will produce a beautifully fluent citation for a claim the paper
never made. Assembling from retrieved text means it can only repeat what was
actually found.

---

### Stage 7 — Check whether any of it worked

Statistics comparing each person to themselves:

- Was their alpha higher after the music than before?
- Did their self-reported stress score drop?
- Do those two agree with each other?
- Does the system improve across the four visits?

We report **effect sizes**, not just "statistically significant." Significance
only says an effect probably isn't zero. Effect size says whether it's big
enough to care about — roughly: 0.2 small, 0.5 medium, 0.8 large.

---

## 6. How we test it: the stranger test

The most important testing decision in the project.

The obvious approach — shuffle everything, train on 80%, test on 20% — is
**wrong here**, because chunks from the same person (often the same session)
land on both sides. The model can effectively recognise the *person* rather than
the *state*, and the score looks great while meaning nothing.

Instead we use **leave-one-subject-out**: train on 19 people, test on the 20th,
repeat 20 times.

That answers the only question that matters clinically: *does this work on
someone it has never met?*

---

## 7. What the results actually say

### The intervention effects — the solid part
*(12 people × 4 sessions = 48 sessions)*

| Measure | Change | Significant? | Effect size |
|---|---|---|---|
| Alpha (relaxation) | **rose** +0.533 | yes, p=0.02 | 0.35 — small |
| Self-reported stress | **fell** −0.90 points | yes, p=0.0002 | 0.58 — medium |
| Theta | fell slightly | borderline, p=0.10 | 0.24 |
| Beta/alpha ratio | fell slightly | no, p=0.38 | 0.13 |

**Brain and self-report agree strongly: r = 0.78.** That's the most reassuring
number here — when the EEG said someone had relaxed, they also *said* they had.
Two independent measurements pointing the same way.

### Personalisation — it does learn

Session 1 (knows nothing about you): **+0.88**
Session 4 (has watched you three times): **+2.19**
→ a **150% improvement**, against a random-choice reference of +0.96.

### Stress detection — real signal, but no winner yet

| | Score |
|---|---|
| Pure guessing would score | 0.33 |
| Classical method (Random Forest) | 0.62 |
| CNN-LSTM | 0.635 |

**What this supports:** both methods find genuine stress information — roughly
double what guessing achieves.

**What this does NOT support:** that the deep model beats the classical one. The
gap (0.015) is about *one sixth* of the variation between individual test
subjects (±0.09). That's noise. It was run on a deliberately small, fast
configuration; the comparison hasn't been run at full scale, and until it has,
claiming the neural network wins would be unfounded.

---

## 8. Three bugs, and why they're worth telling you about

Every one was found by *measuring*, not by reading code. All three produced
output that looked perfectly plausible.

**1. The signal was buried.** Alpha — the thing being measured — was mixed into
the forehead electrodes 10× too weakly, while eye blinks came through at full
strength. Every result came out as exactly zero effect. Nothing crashed. The
code was "working."

**2. The cleaning step deleted the signal.** After fixing #1, alpha and blinks
had become indistinguishable to the blink-removal step, so it removed both —
throwing away 92% of the very thing being studied. Found by measuring the alpha
effect *before* and *after* cleaning and watching it drop from 3.25 to 0.03.

Fix: blinks come from the very front of the head (above the eyes, sensibly);
alpha peaks slightly further back. Separating those two zones let the algorithm
tell them apart. Now **99.8% of the signal survives cleaning**, while blink
contamination drops from 177 to 15 units.

**3. Personalisation measured as zero.** Because the recommender always played
its single best guess, every person heard the same track, so there was no
variety in the data to learn preferences from. It would have reported "no
improvement" forever. Fixed by adding deliberate random exploration.

The lesson worth carrying: **a result of "no effect" is not automatically a
finding. It might be a bug.** All three of these looked like legitimate null
results.

---

## 9. Two honest limitations

### There is no real data yet

The system runs on a **simulator** — synthetic brain recordings with effects
deliberately built in. That's not a substitute for real data, but it isn't
worthless either: because we know exactly what was put in, we can check whether
the analysis finds it. You cannot do that with real recordings.

Three clearly-marked places in the code accept real data when it exists.

### The study design cannot prove the music caused the change

This one matters most.

After a stressful task ends, people calm down **on their own**. The design
measures before-music versus after-music — so the measured improvement is
*music plus natural recovery*, with no way to separate the two.

Two ways to address it:

1. **Add a control condition** — some sessions with silence instead of music.
   The music effect is then the *difference* between them, and natural recovery
   cancels out.
2. **Lean on Stage 4 instead** *(available now, no new data needed)*. If slower
   tempo and drone presence predict *how much* alpha changes, that's evidence
   music matters — because natural recovery has no way of knowing what the tempo
   was. This is the stronger argument, and it's already built.

---

## 10. Glossary

| Term | Plain meaning |
|---|---|
| **EEG** | Recording electrical activity through electrodes on the scalp |
| **Epoch** | One short slice of recording (2 seconds here) |
| **Alpha / Beta / Theta** | Brainwave speed bands; alpha = calm, beta = alert, theta = effort |
| **µV²** (microvolts squared) | The unit of "how much" of a brainwave is present |
| **ICA** | Maths that separates mixed-together sources — used to strip out eye blinks |
| **Spectrogram** | A picture of a signal: time across, frequency up, brightness = amount |
| **CNN** | Image-recognition AI; here it reads spectrograms |
| **LSTM** | AI with a memory, for tracking change over time |
| **Transformer** | AI that weighs which parts of the input matter — and can show you |
| **LOSO** | Leave-one-subject-out: test on a person the model never trained on |
| **PR-AUC** | A score that stays honest when one category is rarer than another |
| **Cohen's d** | How big an effect is: 0.2 small, 0.5 medium, 0.8 large |
| **p-value** | Chance of seeing this if nothing real were happening. Small = probably real |
| **RAG** | Look up real source documents, then answer using them — not from memory |
| **Raga** | A melodic framework in Indian classical music |
| **Drone** | The continuous sustained background note in Indian classical music |
| **Exploration** | Deliberately trying something other than the current best guess, in order to learn |

---

## 11. Running it

```bash
cd ~/ai-lab

# Everything, small and fast (~10 minutes)
uv run python -m neurotune.cli run-all --subjects 6 --sessions 2

# Individual stages
uv run python -m neurotune.cli detect --model transformer
uv run python -m neurotune.cli validate      # did the intervention work?
uv run python -m neurotune.cli map           # what in the music helped?
uv run python -m neurotune.cli recommend --stress 8.2
uv run python -m neurotune.cli explain --stress 8.2

# The full 20 people × 4 sessions (hours)
uv run python -m neurotune.cli run-all --full
```

Small numbers are the default on purpose. The full study is 36,000 slices and 20
rounds of retraining — realistic on this machine, but slow.
