"""
One command to put the whole concept on screen.

    python -m demo.run_demo

Starts the fog node + dashboard on http://localhost:8080 and the 2D lab sim
driving the real edge pipeline, then patrols until you stop it.

To drive the dashboard from Webots instead (higher fidelity, real rendered
frames, optionally the real SSD-MobileNet-V2 weights), start the server alone:

    python -m demo.run_demo --no-sim
    ./scripts/run_webots_demo.sh

and to point a real Jetson Nano at it, run `nano/main_loop.py` on the robot with
`--laptop-url http://<this-machine>:8080`.
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from demo import server
from demo.edge_agent import EdgeAgent
from shared.event_schema import Event


def build_agent(args: argparse.Namespace) -> EdgeAgent:
    """Wire the edge agent straight into the in-process fog pipeline.

    The HTTP hop is skipped here on purpose: this is the single-process
    "everything on one laptop" demo. The Webots controller and a real Nano use
    nano.event_client.EventClient over HTTP against the same POST /event, so the
    edge-fog contract is still the one from Phase 5.
    """

    def on_event(event: Event) -> bool:
        result = server.pipeline.ingest(event)
        server.hub.publish({"kind": "episode", "data": result.to_dict()})
        return True

    def on_telemetry(payload: dict) -> None:
        server.hub.publish({"kind": "telemetry", "data": payload})

    return EdgeAgent(
        on_telemetry=on_telemetry,
        on_event=on_event,
        fps=args.fps,
        time_scale=args.time_scale,
        station_frames=args.station_frames,
        laps=args.laps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Episodic Perception - live browser demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=6.0, help="simulated detector frame rate")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="wall-clock scaling for motor pulses (0 runs as fast as possible)",
    )
    parser.add_argument("--station-frames", type=int, default=90)
    parser.add_argument("--laps", type=int, default=0, help="0 = patrol until stopped")
    parser.add_argument(
        "--no-sim",
        dest="sim",
        action="store_false",
        help="serve the dashboard only; the edge comes from Webots or real hardware",
    )
    parser.add_argument("--log-level", default="info")
    parser.set_defaults(sim=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.sim:
        agent = build_agent(args)
        server.attach_agent(agent)

        original_lifespan = server.app.router.lifespan_context

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan_with_agent(app):
            async with original_lifespan(app):
                agent.start()
                logging.getLogger("demo.run_demo").info(
                    "2D lab sim patrolling. Open http://%s:%d", args.host, args.port
                )
                yield

        server.app.router.lifespan_context = lifespan_with_agent
    else:
        logging.getLogger("demo.run_demo").info(
            "Dashboard only - waiting for an external edge (Webots or Jetson Nano) "
            "to POST to /event and /telemetry."
        )

    uvicorn.run(server.app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
