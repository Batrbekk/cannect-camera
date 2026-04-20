"""
CANNECT.AI -- Synthetic test data sender for dashboard testing.

Sends realistic analytics payloads to the local backend at
http://192.168.0.106:3000/api/analytics/events every 60 seconds, cycling
through 5 cameras and 3 test campaigns with randomised metrics.

Usage:
    python -m src.test_send
    python -m src.test_send --interval 30
    python -m src.test_send --backend http://localhost:3000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
from datetime import datetime, timezone

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("test_send")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = "http://192.168.0.106:3000"
STATION_ID = "69d4e54739079402c7d5608e"

CAMERAS = ["camera_1", "camera_2", "camera_3", "camera_4", "camera_5"]

TEST_CAMPAIGNS = [
    {"campaignId": "camp_nike_summer_2026", "videoId": "vid_nike_001"},
    {"campaignId": "camp_coca_cola_q2", "videoId": "vid_coke_002"},
    {"campaignId": "camp_samsung_galaxy", "videoId": "vid_samsung_003"},
]

# Age bracket probabilities (weighted random)
AGE_GROUPS = ["child", "teen", "young", "adult", "senior"]
AGE_WEIGHTS = [5, 10, 35, 35, 15]  # realistic distribution

GENDERS = ["male", "female", "unknown"]
GENDER_WEIGHTS = [45, 45, 10]


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _random_direction_counts(n_people: int) -> dict:
    """Distribute n_people across direction buckets."""
    to_screen = random.randint(0, n_people)
    remaining = n_people - to_screen
    from_screen = random.randint(0, remaining)
    remaining -= from_screen
    left = random.randint(0, remaining)
    right = remaining - left
    return {
        "toScreen": to_screen,
        "fromScreen": from_screen,
        "left": left,
        "right": right,
    }


def _random_demographics(n_people: int) -> dict:
    """Generate random gender and age group counts."""
    gender_counts = {"male": 0, "female": 0, "unknown": 0}
    age_counts = {"child": 0, "teen": 0, "young": 0, "adult": 0, "senior": 0}
    group_counts = {"individuals": 0, "couples": 0, "families": 0, "largeGroups": 0}

    for _ in range(n_people):
        g = random.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
        gender_counts[g] += 1
        a = random.choices(AGE_GROUPS, weights=AGE_WEIGHTS, k=1)[0]
        age_counts[a] += 1

    # Groups estimation
    group_counts["individuals"] = max(0, n_people - random.randint(0, n_people // 2))
    couples = random.randint(0, n_people // 4)
    group_counts["couples"] = couples
    families = random.randint(0, max(1, n_people // 6))
    group_counts["families"] = families
    group_counts["largeGroups"] = random.randint(0, max(1, n_people // 10))

    with_children = age_counts["child"]

    return {
        "gender": gender_counts,
        "ageGroups": age_counts,
        "groups": group_counts,
        "withChildren": with_children,
    }


def _random_head_angles(n_looking: int) -> dict:
    """Distribute looking count into head angle categories."""
    direct = random.randint(0, n_looking)
    remaining = n_looking - direct
    partial = random.randint(0, remaining)
    glance = remaining - partial
    return {"direct": direct, "partial": partial, "glance": glance}


def _build_traffic_event(n_people: int) -> dict:
    """Build a traffic analytics event."""
    directions = _random_direction_counts(n_people)
    return {
        "type": "traffic",
        "data": {
            "people": {
                "total": n_people,
                "direction": directions,
                "byTimeInterval": {
                    "hour": n_people,
                    "day": n_people * random.randint(8, 16),
                    "week": n_people * random.randint(50, 120),
                },
            },
        },
    }


def _build_attention_event(n_people: int, n_looking: int, n_over5s: int) -> dict:
    """Build an attention analytics event."""
    avg_dwell = round(random.uniform(1.5, 12.0), 2)
    attention_score = min(100, int(
        (avg_dwell / 10.0) * 40
        + (n_looking / max(n_people, 1)) * 30
        + random.uniform(5, 25)
    ))
    unique_viewers = random.randint(max(1, n_looking // 2), n_looking + 1)

    return {
        "type": "attention",
        "data": {
            "peopleAttention": {
                "totalLooking": n_looking,
                "attentionOver5s": n_over5s,
                "averageDwellTime": avg_dwell,
                "attentionScore": attention_score,
                "headAngle": _random_head_angles(n_looking),
                "uniqueViewers": unique_viewers,
            },
        },
    }


def _build_demographic_event(n_people: int) -> dict:
    """Build a demographic analytics event."""
    demographics = _random_demographics(n_people)
    return {
        "type": "demographic",
        "data": demographics,
    }


def _build_engagement_event(n_looking: int) -> dict:
    """Build an engagement analytics event."""
    avg_dwell = round(random.uniform(2.0, 15.0), 2)
    min_dwell = round(random.uniform(0.5, avg_dwell * 0.5), 2)
    max_dwell = round(random.uniform(avg_dwell, avg_dwell * 2.5), 2)
    median_dwell = round(random.uniform(min_dwell, max_dwell), 2)
    viewing_rate = round(random.uniform(0.15, 0.75), 3)
    repeat_viewers = random.randint(0, max(1, n_looking // 3))

    return {
        "type": "engagement",
        "data": {
            "dwellTime": {
                "average": avg_dwell,
                "min": min_dwell,
                "max": max_dwell,
                "median": median_dwell,
            },
            "viewingRate": viewing_rate,
            "repeatViewers": repeat_viewers,
        },
    }


# ---------------------------------------------------------------------------
# Send logic
# ---------------------------------------------------------------------------

async def send_test_batch(
    client: httpx.AsyncClient,
    camera_id: str,
    campaign: dict,
    backend_url: str,
) -> bool:
    """Send a batch of synthetic events for one camera + campaign."""
    n_people = random.randint(3, 25)
    n_looking = random.randint(0, n_people)
    n_over5s = random.randint(0, n_looking)

    events = [
        _build_traffic_event(n_people),
        _build_attention_event(n_people, n_looking, n_over5s),
        _build_demographic_event(n_people),
        _build_engagement_event(n_looking),
    ]

    payload = {
        "stationId": STATION_ID,
        "cameraId": camera_id,
        "campaignId": campaign["campaignId"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }

    url = f"{backend_url}/api/analytics/events"

    try:
        response = await client.post(url, json=payload, timeout=10.0)
        if response.is_success:
            log.info(
                "  [OK %d] %s | campaign=%s | people=%d looking=%d att5s=%d",
                response.status_code,
                camera_id,
                campaign["campaignId"][:20],
                n_people,
                n_looking,
                n_over5s,
            )
            return True
        else:
            log.warning(
                "  [ERR %d] %s | %s",
                response.status_code,
                camera_id,
                response.text[:200],
            )
            return False
    except httpx.ConnectError:
        log.error("  [CONNECT ERROR] Cannot reach %s -- is the backend running?", url)
        return False
    except httpx.TimeoutException:
        log.error("  [TIMEOUT] %s", camera_id)
        return False
    except Exception as exc:
        log.error("  [ERROR] %s: %s", camera_id, exc)
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main(backend_url: str, interval: int) -> None:
    """Run the synthetic data sender in an infinite loop."""
    log.info("=" * 60)
    log.info("CANNECT.AI Test Data Sender")
    log.info("  Backend:    %s", backend_url)
    log.info("  Station:    %s", STATION_ID)
    log.info("  Cameras:    %s", ", ".join(CAMERAS))
    log.info("  Campaigns:  %d", len(TEST_CAMPAIGNS))
    log.info("  Interval:   %ds", interval)
    log.info("=" * 60)

    campaign_index = 0

    async with httpx.AsyncClient() as client:
        batch_num = 0
        while True:
            batch_num += 1
            campaign = TEST_CAMPAIGNS[campaign_index]

            log.info(
                "\n--- Batch #%d | Campaign: %s | %s ---",
                batch_num,
                campaign["campaignId"],
                datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            )

            success_count = 0
            for cam in CAMERAS:
                ok = await send_test_batch(client, cam, campaign, backend_url)
                if ok:
                    success_count += 1

            log.info(
                "--- Batch #%d complete: %d/%d cameras OK ---",
                batch_num,
                success_count,
                len(CAMERAS),
            )

            # Cycle to next campaign every batch
            campaign_index = (campaign_index + 1) % len(TEST_CAMPAIGNS)

            log.info("Sleeping %ds until next batch...\n", interval)
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser(
        description="CANNECT.AI -- Send synthetic analytics data to backend for dashboard testing"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=BACKEND_URL,
        help=f"Backend URL (default: {BACKEND_URL})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between batches (default: 60)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(backend_url=args.backend, interval=args.interval))
    except KeyboardInterrupt:
        log.info("\nStopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    cli()
