# OOD misdiagnosis coding — read me first

The 16 questions in this batch are about real tea pests/diseases that were
deliberately left out of the KG. You get the model answers that did NOT
abstain — 99 rows in `second_annotator_sheet.csv`, one answer per row. For
each one, judge whether the answer would mislead a grower.

Please code on your own: don't look at the first coder's labels, and don't
talk the cases through with anyone until you've finished.

## Columns to fill

1. **label_force_map_coincidental_generic** — pick exactly one:
   - `force-map` — the answer commits to an organism that IS in the KG as its
     diagnosis. A confident wrong name counts even if hedged ("likely 茶毛虫").
   - `coincidental` — the answer actually gives the pest's correct real name
     (out of the model's own knowledge). Still this category if it then adds
     control advice.
   - `generic` — no organism named, only symptoms / general advice.
2. **prescribes_in_kg_pesticide_yes_no** — `yes` only when it positively
   recommends a named product that is on the in-KG control list. "Spray a
   fungicide" with no product is `no`; natural enemies (瓢虫, 草蛉, 捕食螨…)
   don't count as pesticides; watch for variant spellings of the same
   product (甲基托布津 = 甲基硫菌灵, 恶霉灵 = 噁霉灵, Bt = 苏云金杆菌).
3. **annotator_notes** — jot down what you had to think about: aliases that
   point at a KG node, made-up names that point at nothing, name
   normalisations. Mark anything you're unsure about with `?`. The obvious
   rows can stay blank.

## Things to watch

- Code from the response text. The `true_unseen_pest` column is there so you
  can see what the answer SHOULD have said — don't diagnose from it.
- Giving the correct name in passing but diagnosing something else is still
  `force-map`; what matters is the diagnosis the answer actually commits to.
- Rows that only abstained or deferred to an extension station with no
  diagnosis have already been removed (an answer that diagnoses and THEN
  says "confirm with the station" stays in).

Send the filled CSV back when done. `compute_agreement.py` works out raw
agreement and Cohen's κ against the first coder on the items both of you
coded, and prints the disagreement list for adjudication.
