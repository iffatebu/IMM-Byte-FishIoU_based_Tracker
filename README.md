# IMM-Byte: Identity-Consistent Fish Tracking for Underwater Images
An enhanced version of ByteTrack for tracking fish in underwater low-light, occlusion based images. IMM-Byte replaces ByteTrack's single motion model with an Interacting Multiple Model (IMM) filter (Constant Velocity + Constant Acceleration) and introduces FishIoU, an association metric adapted for underwater occlusions, so that individual fish keep the same identity as they swim past camera obstructions, accelerate suddenly, or turn sharply.

Underwater survey cameras record fish swimming past structural bars, vegetation, and each other. Standard tracking software often "loses" a fish mid-video and assigns it a new identity when it reappears which is called an identity switch (IDSW). If a fish is double-counted this way, abundance estimates for fisheries management become unreliable. IMM-Byte reduces this problem substantially: compared to the widely used ByteTrack tracker, it cuts identity switches by 47.84% (from 485 to 253) on our GFISHERD24 survey dataset, while also improving overall tracking accuracy.


| Header 1 | Header 2 | Header 3 |
| --- | --- | --- |
| Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
| Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |

