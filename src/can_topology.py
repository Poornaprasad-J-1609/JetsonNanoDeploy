#!/usr/bin/env python3
"""CAN adapter topology helpers for one, two, or four RobStride USB-CAN buses."""

import os

from robstride_can_interface import ATUsbCan


CAN_TOPOLOGY_BUS_NAMES = {
    1: ("can0",),
    2: ("front", "back"),
    4: ("FR", "FL", "BR", "BL"),
}


def add_can_topology_args(parser, default_port="/dev/ttyUSB0", default_can_count=2):
    parser.add_argument(
        "--can-count",
        type=int,
        choices=sorted(CAN_TOPOLOGY_BUS_NAMES),
        default=int(default_can_count),
        help=(
            "number of RobStride USB-CAN adapters: "
            "1=all joints on one adapter, 2=front/back, 4=FR/FL/BR/BL"
        ),
    )
    parser.add_argument(
        "--can-ports",
        nargs="*",
        default=None,
        help=(
            "ordered USB-CAN ports. Order: 1 CAN -> can0; "
            "2 CAN -> front back; 4 CAN -> FR FL BR BL"
        ),
    )
    parser.add_argument(
        "--port",
        default=default_port,
        help="legacy fallback USB-CAN port used when topology-specific ports are not given",
    )
    parser.add_argument(
        "--port-can0",
        default=None,
        help="USB-CAN port for one-CAN topology",
    )
    parser.add_argument(
        "--port-can1",
        default=None,
        help="generic second USB-CAN port fallback",
    )
    parser.add_argument(
        "--port-can2",
        default=None,
        help="generic third USB-CAN port fallback",
    )
    parser.add_argument(
        "--port-can3",
        default=None,
        help="generic fourth USB-CAN port fallback",
    )
    parser.add_argument(
        "--port-front",
        default=None,
        help="USB-CAN port for front legs in two-CAN topology; overrides --port",
    )
    parser.add_argument(
        "--port-back",
        default=None,
        help="USB-CAN port for back legs in two-CAN topology; overrides --port",
    )
    parser.add_argument(
        "--port-fr",
        default=None,
        help="USB-CAN port for FR leg in four-CAN topology",
    )
    parser.add_argument(
        "--port-fl",
        default=None,
        help="USB-CAN port for FL leg in four-CAN topology",
    )
    parser.add_argument(
        "--port-br",
        default=None,
        help="USB-CAN port for BR leg in four-CAN topology",
    )
    parser.add_argument(
        "--port-bl",
        default=None,
        help="USB-CAN port for BL leg in four-CAN topology",
    )


def bus_names_for_count(can_count):
    can_count = int(can_count)
    if can_count not in CAN_TOPOLOGY_BUS_NAMES:
        raise ValueError(f"Unsupported CAN count {can_count}; expected 1, 2, or 4")
    return list(CAN_TOPOLOGY_BUS_NAMES[can_count])


def joint_leg(joint_name):
    return str(joint_name).split("_", 1)[0]


def joint_bus_for_count(joint_name, can_count):
    leg = joint_leg(joint_name)
    if can_count == 1:
        return "can0"
    if can_count == 2:
        if leg in ("FR", "FL"):
            return "front"
        if leg in ("BR", "BL"):
            return "back"
    if can_count == 4 and leg in ("FR", "FL", "BR", "BL"):
        return leg
    raise ValueError(f"Cannot infer CAN bus for joint '{joint_name}' with can_count={can_count}")


def resolve_joint_can_bus(policy_order, can_count):
    return {
        joint_name: joint_bus_for_count(joint_name, int(can_count))
        for joint_name in policy_order
    }


def _ports_from_ordered_list(can_count, can_ports):
    if can_ports is None:
        return None
    ports = [str(port) for port in can_ports if str(port).strip()]
    bus_names = bus_names_for_count(can_count)
    if len(ports) != len(bus_names):
        raise ValueError(
            f"--can-count {can_count} requires exactly {len(bus_names)} --can-ports "
            f"({', '.join(bus_names)}), got {len(ports)}"
        )
    return dict(zip(bus_names, ports))


def resolve_port_by_bus(args):
    can_count = int(args.can_count)
    ordered = _ports_from_ordered_list(can_count, getattr(args, "can_ports", None))
    if ordered is not None:
        return ordered

    port = str(getattr(args, "port", "/dev/ttyUSB0"))
    if can_count == 1:
        return {
            "can0": getattr(args, "port_can0", None) or port,
        }
    if can_count == 2:
        return {
            "front": getattr(args, "port_front", None) or getattr(args, "port_can0", None) or port,
            "back": getattr(args, "port_back", None) or getattr(args, "port_can1", None) or port,
        }
    if can_count == 4:
        front_fallback = getattr(args, "port_front", None) or port
        back_fallback = getattr(args, "port_back", None) or port
        return {
            "FR": getattr(args, "port_fr", None) or getattr(args, "port_can0", None) or front_fallback,
            "FL": getattr(args, "port_fl", None) or getattr(args, "port_can1", None) or front_fallback,
            "BR": getattr(args, "port_br", None) or getattr(args, "port_can2", None) or back_fallback,
            "BL": getattr(args, "port_bl", None) or getattr(args, "port_can3", None) or back_fallback,
        }
    raise ValueError(f"Unsupported CAN count {can_count}; expected 1, 2, or 4")


def ports_for_active_joints(port_by_bus, joint_can_bus, active_joints):
    used_bus_names = {
        joint_can_bus[joint_name]
        for joint_name in active_joints
        if joint_name in joint_can_bus
    }
    return {
        bus_name: port
        for bus_name, port in port_by_bus.items()
        if bus_name in used_bus_names
    }


def physical_port_identity(port):
    """Return a stable identity for duplicate-port checks."""
    return os.path.realpath(str(port))


def validate_unique_motor_ids_per_physical_bus(
    motor_ids,
    joint_can_bus,
    active_joints,
    port_by_bus,
):
    """
    Reject duplicate motor IDs that would be addressed on the same physical CAN adapter.

    Duplicate motor IDs are valid only when they live on different CAN networks. If two
    logical buses point to the same /dev/ttyUSB* and reuse an ID, RobStride set-zero,
    stop, enable, and MIT packets cannot target one motor independently.
    """
    seen = {}
    conflicts = []

    for joint_name in active_joints:
        if joint_name not in motor_ids:
            continue
        bus_name = joint_can_bus.get(joint_name)
        if bus_name is None:
            continue
        port = port_by_bus.get(bus_name)
        if port is None:
            continue

        key = (physical_port_identity(port), int(motor_ids[joint_name]))
        if key in seen:
            other_joint, other_bus, other_port = seen[key]
            conflicts.append(
                f"{other_joint} [{other_bus}:{other_port}] and "
                f"{joint_name} [{bus_name}:{port}] both use motor_id=0x{key[1]:02X}"
            )
        else:
            seen[key] = (joint_name, bus_name, port)

    if conflicts:
        raise ValueError(
            "Duplicate RobStride motor IDs share the same physical CAN adapter. "
            "Use unique IDs on one CAN bus, or put reused IDs on separate adapters. "
            "Conflicts: " + "; ".join(conflicts)
        )


def open_can_buses(port_by_bus, baud, timeout=None):
    buses = {}
    opened_by_port = {}
    try:
        for bus_name, port in port_by_bus.items():
            if port in opened_by_port:
                buses[bus_name] = opened_by_port[port]
                continue

            kwargs = {"port": port, "baud": baud}
            if timeout is not None:
                kwargs["timeout"] = timeout
            adapter = ATUsbCan(**kwargs).open()
            opened_by_port[port] = adapter
            buses[bus_name] = adapter
    except Exception:
        close_can_buses(buses)
        raise
    return buses


def close_can_buses(buses):
    closed_ids = set()
    for bus in buses.values():
        if id(bus) in closed_ids:
            continue
        bus.close()
        closed_ids.add(id(bus))


def topology_lines(can_count, port_by_bus):
    lines = [f"CAN topology: {int(can_count)} adapter(s)"]
    for bus_name in bus_names_for_count(can_count):
        if bus_name in port_by_bus:
            lines.append(f"  {bus_name:5s} -> {port_by_bus[bus_name]}")
    return lines
