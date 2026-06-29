"""
Composition Reconstruction
==========================

Sprint 1:
Role Analyzer

Input:
    ["omen", "viper", "cypher", ...]

Output:
{
    "controller": {
        "count": 2,
        "agents": ["omen", "viper"]
    },
    ...
}
"""
import math

from training_v2.config import AGENT_ROLE_MAP

ROLE_ORDER = [
    "duelist",
    "initiator",
    "controller",
    "sentinel",
]

def analyze_roles(
    agent_frequency: dict[str, int]
) -> dict:
    """
    Analyze role distribution using historical
    agent played frequency.

    Parameters
    ----------
    agent_frequency

        Example

        {
            "omen": 14,
            "viper": 8,
            "sova": 12
        }

    Returns
    -------
    dict
    """

    role_analysis = {

        role: {

            "agent_count": 0,

            "total_frequency": 0,

            "agents": {}

        }

        for role in ROLE_ORDER

    }

    unknown_agents = []

    for agent, frequency in agent_frequency.items():

        role = AGENT_ROLE_MAP.get(agent)

        if role is None:

            unknown_agents.append(agent)

            continue

        role_analysis[role]["agent_count"] += 1

        role_analysis[role]["total_frequency"] += frequency

        role_analysis[role]["agents"][agent] = frequency

    if unknown_agents:

        raise ValueError(

            f"Unknown agents: {unknown_agents}"

        )

    return role_analysis

def calculate_role_distribution(
    role_analysis: dict
) -> dict:
    """
    Calculate role distribution
    based on historical frequency.
    """

    total_frequency = sum(
        role["total_frequency"]
        for role in role_analysis.values()
    )

    if total_frequency == 0:
        raise ValueError(
            "Total role frequency cannot be zero."
        )

    distribution = {}

    for role in ROLE_ORDER:

        role_frequency = role_analysis[role]["total_frequency"]

        distribution[role] = {

            "agent_count":
                role_analysis[role]["agent_count"],

            "total_frequency":
                role_frequency,

            "ratio":
                role_frequency / total_frequency,

            "agents":
                role_analysis[role]["agents"]

        }

    return distribution

ROLE_SLOT_CAP = {
    "controller": 2,
    "initiator": 2,
    "duelist": 2,
    "sentinel": 2,
}

TARGET_COMPOSITION_SIZE = 5

def allocate_role_slots(
    distribution: dict
) -> dict:
    """
    Allocate 5 composition slots using
    Largest Remainder Method + Role Cap.
    """

    allocation = {}

    remainders = []

    total_floor = 0

    # -----------------------------------------
    # Step 1
    # Raw Slot
    # -----------------------------------------

    for role in ROLE_ORDER:

        ratio = distribution[role]["ratio"]

        raw_slot = ratio * TARGET_COMPOSITION_SIZE

        floor_slot = math.floor(raw_slot)

        floor_slot = min(
            floor_slot,
            ROLE_SLOT_CAP[role]
        )

        allocation[role] = floor_slot

        total_floor += floor_slot

        remainders.append(

            (

                role,

                raw_slot - floor_slot

            )

        )

    # -----------------------------------------
    # Step 2
    # Remaining Slot
    # -----------------------------------------

    remaining = (

        TARGET_COMPOSITION_SIZE

        -

        total_floor

    )

    remainders.sort(

        key=lambda x: x[1],

        reverse=True

    )
    
    DEBUG = False
    
    if DEBUG:
    
        print("\nRemainders")
        pprint(remainders)

        print("\nAllocation")
        pprint(allocation)

        print(f"Remaining = {remaining}")

    while remaining > 0:

        allocated = False

        for role, _ in remainders:

            if allocation[role] >= ROLE_SLOT_CAP[role]:
                continue

            allocation[role] += 1

            remaining -= 1

            allocated = True

            if remaining == 0:
                break

        if not allocated:
            break

    return allocation

def select_agents(
    distribution: dict,
    slots: dict,
) -> dict:
    """
    Select final composition based on
    allocated role slots.

    Parameters
    ----------
    distribution

    slots

    Returns
    -------
    dict
    """

    composition = []

    for role in ROLE_ORDER:

        role_slots = slots[role]

        if role_slots == 0:
            continue

        agents = distribution[role]["agents"]

        sorted_agents = sorted(
            agents.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected = [

            agent

            for agent, _ in sorted_agents[:role_slots]

        ]

        composition.extend(selected)
        
    if len(composition) != TARGET_COMPOSITION_SIZE:

        raise ValueError(

            "Composition reconstruction failed. "
            f"Expected {TARGET_COMPOSITION_SIZE} agents, "
            f"got {len(composition)}."

        )

    return {

        "composition": composition,

        "slots": slots,

        "distribution": distribution

    }

if __name__ == "__main__":

    sample = {

        "omen":14,

        "jett":14,

        "sova":14,

        "kayo":10,

        "killjoy":8,

        "cypher":6,

        "deadlock":4

    }
    

    from pprint import pprint

    analysis = analyze_roles(sample)

    distribution = calculate_role_distribution(
        analysis
    )

    print()

    print("=" * 60)
    print("ROLE DISTRIBUTION")
    print("=" * 60)

    pprint(distribution)
    
    slots = allocate_role_slots(
        distribution
    )

    print()

    print("=" * 60)
    print("ROLE SLOT")
    print("=" * 60)

    pprint(slots)
    
    result = select_agents(
        distribution,
        slots
    )

    print()

    print("=" * 60)
    print("FINAL COMPOSITION")
    print("=" * 60)

    pprint(result)

