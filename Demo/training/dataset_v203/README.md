# Dataset Snapshot v203

## Summary
- **Accepted Clean Images:** 1,092 (Train: 882, Val: 142, Test: 68 locked)
- **Total Clean Boxes:** 3,578 (Class 0 Motorcycle: 1,933, Class 1 Helmet: 1,062, Class 2 No Helmet: 583)
- **Pending Images:** 25 (Held in group_0000 to respect split lock)
- **Excluded Duplicates:** 1,136 (1,132 Roboflow augmented variants + 4 earlier duplicates)
- **Goal Target:** 5,000 clean images
- **Progress:** 1,092 / 5,000 (21.84%)
- **Remaining to Goal:** 3,908 images

## Audit of 1,492 Pending Items
All 1,492 pending items from DataSet2 were audited in this batch:
1. 335 unique canonical frames selected and added (279 to train, 56 to val).
2. 1,132 redundant augmented variants (.rf.<hash>) excluded as duplicates.
3. 25 rows in group_0000 held pending to protect the locked test split (68 images).
