from pandapower.create import (
    create_empty_network,
    create_bus,
    create_ext_grid,
    create_load,
    create_line,
    create_transformer,
    create_switch
)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def build_base_net():
    net = create_empty_network()

    #create buses
    bus1 = create_bus(net, vn_kv=20., name="Bus 1")
    bus2 = create_bus(net, vn_kv=0.4, name="Bus 2")
    bus3 = create_bus(net, vn_kv=0.4, name="Bus 3")

    #create bus elements
    ext = create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    load = create_load(net, bus=bus3, p_mw=0.05, q_mvar=0.005, name="Load")

    #create branch elements
    trafo = create_transformer(net, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
    line = create_line(net, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")

    idx = {
        "bus1": bus1, "bus2": bus2, "bus3": bus3,
        "ext": ext, "load": load, "trafo": trafo, "line": line,
    }
    return net, idx


def build_110kV_net():
    net = create_empty_network()

    #create buses
    bus1 = create_bus(net, vn_kv=110., name="Bus 1")
    bus2 = create_bus(net, vn_kv=20., name="Bus 2")
    bus3 = create_bus(net, vn_kv=0.4, name="Bus 3")
    bus4 = create_bus(net, vn_kv=0.4, name="Bus 4")

    #create bus elements
    ext = create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    load = create_load(net, bus=bus4, p_mw=0.05, q_mvar=0.005, name="Load")

    #create branch elements
    trafo = create_transformer(net, hv_bus=bus1, lv_bus=bus2, std_type="25 MVA 110/20 kV", name="Trafo 1")
    trafo = create_transformer(net, hv_bus=bus2, lv_bus=bus3, std_type="0.4 MVA 20/0.4 kV", name="Trafo 2")
    line = create_line(net, from_bus=bus3, to_bus=bus4, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")

    idx = {
        "bus1": bus1, "bus2": bus2, "bus3": bus3, "bus4": bus4,
        "ext": ext, "load": load, "trafo": trafo, "line": line,
    }
    return net, idx

def build_2_load_net():
    net = create_empty_network()

    #create buses
    bus1 = create_bus(net, vn_kv=20., name="Bus 1")
    bus2 = create_bus(net, vn_kv=0.4, name="Bus 2")
    bus3 = create_bus(net, vn_kv=0.4, name="Bus 3")

    #create bus elements
    ext = create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    load1 = create_load(net, bus=bus3, p_mw=0.05, q_mvar=0.005, name="Load 1")
    load2 = create_load(net, bus=bus3, p_mw=0.02, q_mvar=0.005, name="Load 2")

    #create branch elements
    trafo = create_transformer(net, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
    line = create_line(net, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")

    idx = {
        "bus1": bus1, "bus2": bus2, "bus3": bus3,
        "ext": ext, "load1": load1, "load2": load2, "trafo": trafo, "line": line,
    }
    return net, idx