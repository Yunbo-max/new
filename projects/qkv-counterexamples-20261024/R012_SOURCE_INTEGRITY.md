# R012 source integrity

Authoritative parent Git SHA: 4683d005f0b01b542961ce0dbef88502c02948ed. The original local `R012_cpu_checks.py` and `R012_metrics.json` were transmitted as Git blobs **with identical SHA-1 Git blob hashes** to files verified in the CPU runtime. Code blob: `0201ada159831db5384f2f92e63616fdbc6715ed`, metrics blob: `f8fa832bc12983f07c9623b342b3e1505fcb11b0`. The CPU test was executed successfully twice and stdout matched exactly. This check is binary equivalence, NOT formal mathematical review or native model reproducibility. Independent reviewer still pending.
