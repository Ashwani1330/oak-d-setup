- test pick and place of single-object
- method to verify if the depth frames are accurately coming
- method to check the angles are accurately measured.
- fix cross-clock latency (~5sec)


Identified issue:
- Axis is not coming proper (segmented area 3d pcl going down deep into the table)
	- some depth data issue
	- 		


---


## Anti-backlog protections:

On the **Jetson sender**:

* both DepthAI output queues are created with `maxSize=2, blocking=False`, so they cannot grow unbounded
* for both video and depth, you call `tryGet()` and then drain the queue in a loop, keeping only the **newest** frame before processing/sending it 

So on the sender side, the policy is basically:

**drop old frames, keep latest**

On the **server remote receiver**:

* `rgb_q` and `pcl_q` are custom `_LatestMessageQueue` objects that store only **one message total**; every new `put()` overwrites the old one
* the RGB history used for pairing is a `deque(maxlen=max_rgb_buffer)`, default `8`, so that buffer is also bounded
* partial UDP depth frames in `_pending_depth` are cleaned by age using `stale_depth_ms`, default `120 ms`, so incomplete chunk assemblies do not live forever
* on startup/reset, `start()` clears the RGB history, latest queues, and pending depth state
* the RTSP capture also tries to keep buffering low with `CAP_PROP_BUFFERSIZE = 1` plus FFmpeg `nobuffer/low_delay` options 

So overall: **yes, there is safety against queues filling up**. The design is intentionally lossy rather than backlog-preserving.

The main remaining weak spot is not queue explosion, but this:

* `_pending_depth` is only bounded by **age**, not by an explicit max-count
* if the network gets very bad and many partial depth sequences arrive within the stale window, it can still grow temporarily, though only for that short window 

So practical verdict:

* for normal operation: **safe enough**
* behavior under stress: **drops frames instead of building latency**
* only minor improvement I’d consider later: add a small hard cap on `_pending_depth` entries, for example 16 or 32 total seqs, and evict oldest when exceeded

That would make it more robust, but it is not the first thing you need to fix.

---

### Observed:

- when the tray is not used and objects are directly kept on the gray table, axis detection has been observed to be bad.

- Good contrast help (blue crate) in good axis detection.

- two or more objects close together getting segemented as one: most probably not the wireless-feed issue but rather a thing to be solved in the multi-object sam-3 pipeline.

---

### Scientific data-driven metric capture:

- read fps (from cam direct) : cam -> jetson - jetson
- write fps : jetson -> network - jetson
- read fps : network -> server pc - pc


- **cuda utilization** in compression and decompression

fps: 
which is the bottleneck?: compression-decompression | network


data-driven pipeline:
	- wired (ethernet)
	- wireless (wi-fi)

cpu & gpu usage

---

### Wht zstd not on Jetson GPU?
- Non-trivial technically: new library/API, CUDA buffers, chunk metadata, stream management.
- Non-trivial algorithmically: Zstd has stateful entropy coding and block dependencies.
- Non-trivial practically on Nano: transfer/setup overhead may cancel the win for your per-frame pipeline.

So “GPU Zstd” is not wrong. It is just not the first optimization I would bet on for this sender.

- test --depth-comp none on wired Ethernet,
- test lz4 against zstd,
- reduce packet count / Python send overhead,
- and only then consider a CUDA/nvCOMP rewrite.

---

### Better algos and methods:

- default to lz4
- benchmark against none
- only build RVL if LZ4/none is still not good enough.

---

