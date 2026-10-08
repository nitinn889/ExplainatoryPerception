# Running the demo — start here

For someone who has never touched this project before. No prior knowledge
assumed. Follow it top to bottom.

**What you are about to show:** a robot that patrols a room, notices when it can
only half-see an object, physically repositions itself until it can see the
whole thing, remembers what it saw as a sentence, notices when something has
been moved, and answers questions about any of it.

**Time needed:** about 5 minutes the first time, 30 seconds after that.

---

# Part 1 — Get it running

## Step 1: Check you have Python

Open a terminal (on Windows, use **PowerShell**) and type:

```bash
python3 --version
```

> **Windows:** use `python --version` instead. Everywhere below that says
> `python3`, you type `python`.

You need **3.10 or higher**. If you see `3.10.x`, `3.12.x`, `3.14.x` — you are
fine.

<details>
<summary>If it says "command not found"</summary>

Python is not installed. Get it from [python.org/downloads](https://www.python.org/downloads/).
On Windows, **tick "Add Python to PATH"** on the first screen of the installer,
or nothing below will work.
</details>

## Step 2: Download the project

```bash
git clone -b claude/wizardly-noether-uz83tj https://github.com/nitinn889/ExplainatoryPerception.git
cd ExplainatoryPerception
```

Takes about a second. The `-b claude/...` part matters — it picks the branch
with the demo on it.

**Check it worked.** Type `ls` (`dir` on Windows). You should see folders named
`demo`, `nano`, `laptop`, `webots`. If you don't, you are in the wrong place.

> Every remaining command must be run **from inside this folder**. If you close
> the terminal and come back, `cd` into it again first.

## Step 3: Make a sandbox for the project's code libraries

```bash
python3 -m venv .venv
```

Takes ~3 seconds and prints nothing. This keeps the project's libraries
separate from the rest of your computer.

Now switch into it:

```bash
source .venv/bin/activate
```

> **Windows:** `.venv\Scripts\activate`

Your prompt should now start with `(.venv)`. **If it doesn't, stop and fix this
before continuing** — the next steps will install things in the wrong place.

<details>
<summary>If you get "ensurepip is not available" (Ubuntu/Debian)</summary>

```bash
sudo apt install python3-venv
```
then run the `python3 -m venv .venv` command again.
</details>

## Step 4: Install the libraries

**The quick way — about 15 seconds. Recommended.**

```bash
pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" \
            "requests>=2.31" "numpy>=1.24" "opencv-python-headless>=4.8"
```

> **Windows:** remove the `\` characters and put it all on one line.

**The full way — 1 to 3 GB, 10+ minutes.** Only if you have time and a good
connection:

```bash
pip install -r requirements-laptop.txt -r requirements-nano.txt
```

**What is the difference?** The full version adds ChromaDB (a real vector
database) and a language model for understanding the meaning of your questions.
The quick version uses simpler built-in substitutes. **The demo looks and works
the same either way** — it just matches questions on keywords rather than
meaning. The dashboard displays which one is active, so you can say so honestly
if asked.

You can start with the quick version today and install the full one later.

**Optional — if you want to run the tests too:**

```bash
pip install -r requirements-dev.txt
```

## Step 5: Run it

```bash
python -m demo.run_demo
```

Within a couple of seconds you should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
2026-10-08 12:58:09 [INFO] demo.edge_agent: PATROL -> ADJUST  signal=turn_right  dets=3
2026-10-08 12:58:09 [INFO] demo.edge_agent: ADJUST -> CONFIRM  signal=fully_visible  dets=3
2026-10-08 12:58:10 [INFO] demo.edge_agent: CONFIRM -> LOGGED  signal=fully_visible  dets=3
```

### ⚠️ The most common confusion

**The terminal will not go back to the normal prompt, and messages keep
scrolling. That is correct. It is working.** That is the robot reporting what it
is doing. Do not close it, do not press Ctrl+C.

Leave that terminal alone and open a web browser:

## 👉 **http://localhost:8080**

The dashboard appears and the robot starts patrolling on its own.

To stop everything later: click the terminal and press **Ctrl+C**.

---

# Part 2 — Before you present

Do these in the 5 minutes beforehand.

- [ ] **Start it early.** Let it run for at least a minute before you talk, so
      the memory panel on the right already has entries. Starting from empty is
      a weak opening.
- [ ] **Full-screen the browser** (press **F11**). The dashboard has three
      columns and needs the width.
- [ ] **Check the projector** shows the whole page, especially the right-hand
      column.
- [ ] **Have the backup video** on your laptop in case the projector or wifi
      misbehaves.
- [ ] **Read Part 4** so no question catches you out.

---

# Part 3 — What to show, in order

About five minutes of talking. Each heading is roughly one thing to say.

### 1. "Here is the overall design"

Point at the grey line under the title:

> `Edge: camera → detect → scene graph → perception–action loop → importance filter`
> `Fog: caption → embed → vector memory → RAG`

Say: the left half is meant to run on the robot itself (a Jetson Nano), the
right half on a laptop, and they talk to each other over the network. The demo
runs both on one computer, but it is the same code.

### 2. "The robot is really looking at things"

Point at the **Camera view** panel, top left. Every coloured box is a real
object, worked out by projecting its actual position through a camera lens
model — not a pre-made animation.

The **red stripes** down the left and right edges are the warning zone. A box
touching one of those means "I can only half-see this".

### 3. ⭐ The main idea — spend the most time here

Watch the location label (top-right of the camera panel) until it says
**`lab_desk_3`**. You will see:

1. The blue bottle sits inside the **right-hand red stripe** — half out of view
2. The state badge changes to **ADJUST**
3. `turn_right` appears at the bottom of the camera view — the robot is turning
4. The bottle **slides into the middle** of the frame
5. The badge changes to **CONFIRM**, then **LOGGED**

Say this bit clearly:

> It holds **CONFIRM** for four frames in a row before writing anything down.
> That stops it recording a half-corrected view by mistake.

The other two stops correct in different ways — worth pointing out:

| Stop | Problem | What the robot does |
|---|---|---|
| `lab_desk_3` | bottle too far right | turns right |
| `lab_desk_1` | laptop too small / far away | drives forward |
| `lab_shelf_2` | bottle too far left | turns left |

### 4. "It writes down facts, not pictures"

Right-hand column, **Memory**. An entry appears saying something like
*"A bottle is on the dining table"* with a location tag — a sentence, not a
picture or a set of coordinates.

Point at **FRAMES → EPISODES**. It will show something like **130×**. Say: the
robot looked at 800-odd frames and chose to remember about 6 things. That gap is
the filtering doing its job.

### 5. "It notices when someone moves something"

Click the **Move the bottle (staged event)** button. You are playing the person
who moves something while the robot is in another part of the room.

Next time the robot reaches the shelf, the new entry is marked **moved** —
rather than being stored as an unrelated new fact in a different place.

**Be upfront:** say the move is staged. It is the simulated stand-in for the
"staged object movement" experiments the project plan calls for.

### 6. "It doesn't store the same thing twice"

On a later lap you will see an entry showing **2×** or **3×** with a duration
instead of a second identical row. One lasting fact, not one row per glance.

### 7. Ask it questions

Bottom right. Click a suggestion or type your own:

- **"where is the bottle?"** → tells you where it last saw it, and mentions if
  it moved
- **"did anything move?"** → finds the moved entry
- **"what is on desk 3?"** → the right desk

Point at the millisecond timing, and at the list of entries underneath the
answer. Say: the answer is built only from those entries, so it cannot make
anything up.

---

# Part 4 — Questions you will be asked

**"Is this using YOLO?"**
No. It uses SSD-MobileNet-V2, which is what the project requirements specify.
There is no YOLO code anywhere in the project.

**"Are those real AI detections on screen?"**
**Say this before you are asked — it lands much better volunteered.** No. The
demo uses the simulator's own knowledge of where objects are. That is
deliberate: it makes the demo identical every time, and it keeps the focus on
the repositioning behaviour rather than on detector mistakes.

The real detector *is* built in and does work — on real photographs it scores
0.99 on a person and 0.98 on a sports ball. But on this simulated room it finds
only one thing: the desk, which it calls a "bench". The room is made of plain
untextured shapes, and the AI was trained on real photographs, so a plain blue
cylinder does not look like a bottle to it. That is about the graphics, not
about the pipeline. The proof is in `docs/webots_demo_results.md`.

To show the detector working properly, point it at a webcam:
`python -m nano.main_loop --mode usb --display`

**"Is it navigating the room / using SLAM?"**
No, and that is on purpose — the project plan explicitly rules it out. The robot
follows a fixed list of stops, each with a simple location label. The part that
sees and remembers never gets told where the robot is.

**"Has it been tested?"**
Yes. Run it in front of them if you like:

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

On the quick install you will see **39 passed, 2 skipped**; on the full install,
**42 passed, 1 skipped**. The skips are tests that need the optional extras —
they skip rather than pretending to pass. Measured demo results are in
`docs/webots_demo_results.md`.

**"What doesn't work?"**
There is an honest list in `demo/README.md` under *Honest limitations*. Read it
beforehand. Short version: the simple simulator has no real images, no objects
blocking other objects, the patrol route is fixed, the object moves are staged,
and very flat objects (like a book lying down) confuse the "too far away" rule.

**"Can I see the proper 3D simulator?"**
Yes, if Webots is installed — see Part 6.

---

# Part 5 — If something goes wrong

| What you see | What it means | Fix |
|---|---|---|
| `No module named demo` | wrong folder | `cd` into the `ExplainatoryPerception` folder |
| `Could not open requirements file` | wrong folder | same as above |
| `Address already in use` | something else is on port 8080 | `python -m demo.run_demo --port 8090`, then open `localhost:8090` |
| Browser says "can't connect" | server not running | check the terminal for a red error |
| Dashboard loads but is empty | just started | wait ~20 seconds |
| Nothing moving, says "reconnecting" | server stopped | restart with `python -m demo.run_demo` |
| `source: command not found` | you are on Windows | use `.venv\Scripts\activate` |
| `pip: command not found` | sandbox not switched on | redo Step 3; look for `(.venv)` in your prompt |
| First run hangs for minutes | the full install is downloading its model | let it finish once; it is cached afterwards |

**Golden rule:** almost every problem is "wrong folder" or "forgot to activate
the sandbox". Check both first.

---

# Part 6 — Optional: the 3D simulator (Webots)

Only if your teacher asks, and only if you set it up beforehand. Install Webots
R2023b or newer from [cyberbotics.com](https://cyberbotics.com/) — about 1 GB.

You need **two terminals**, both inside the project folder with the sandbox
activated.

Terminal 1:
```bash
python -m demo.run_demo --no-sim
```

Terminal 2:
```bash
./scripts/run_webots_demo.sh
```

Webots opens and shows the robot driving around a 3D room. The **same
dashboard** at `localhost:8080` fills in as before — but now the camera panel
shows what the 3D camera genuinely sees.

If Webots is black or extremely slow, your graphics drivers are the problem.
Use the browser demo instead; it needs no graphics hardware at all.

---

# Quick reference card

Print this bit.

```bash
# every time, from inside the ExplainatoryPerception folder:
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m demo.run_demo
# then open http://localhost:8080
# Ctrl+C in the terminal to stop
```

**Talking order:** architecture → camera is real → **the bottle correction** →
memory stores sentences → click *Move the bottle* → compression (2×, 3×) → ask
it a question.

**Say early:** "the detections are ground truth from the simulator, not live AI
output — here's why."
