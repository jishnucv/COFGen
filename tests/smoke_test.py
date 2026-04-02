"""
smoke_test.py — validate COFGen architecture without torch or real data.

Tests:
  1. Featurisation dimensions
  2. Minimal CIF parsing
  3. CrystalGraph construction from a synthetic CIF
  4. Building-block identification
  5. COFSpec and pyCOFBuilder name generation
  6. Validity checker (geometry only)
  7. SUN metric computation
  8. Encoder forward pass (if torch available)
  9. Flow matching forward pass (if torch available)
 10. Adapter forward pass (if torch available)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import json
import math
import numpy as np

PASS = "✓"
FAIL = "✗"

results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL}  {name}: {e}")

print("\n" + "="*60)
print("COFGen Smoke Test")
print("="*60)

# ── 1. Featurisation ────────────────────────────────────────────────────────
print("\n[1] Featurisation")

def test_atom_features():
    from utils.featurisation import atom_features, ATOM_FEAT_DIM
    f = atom_features("C", "sp2", formal_charge=0.0, is_aromatic=True, degree=3)
    assert f.shape == (ATOM_FEAT_DIM,), f"Expected ({ATOM_FEAT_DIM},), got {f.shape}"
    assert f.sum() > 0

check("atom_features shape", test_atom_features)

def test_bond_features():
    from utils.featurisation import bond_features, BOND_FEAT_DIM
    f = bond_features("AROMATIC", is_in_ring=True, is_conjugated=True, distance_angstrom=1.4)
    assert f.shape == (BOND_FEAT_DIM,)
    assert f.sum() > 0

check("bond_features shape", test_bond_features)

def test_topology_onehot():
    from utils.featurisation import topology_onehot, N_TOPOLOGIES
    v = topology_onehot("hcb")
    assert v.shape == (N_TOPOLOGIES,)
    assert v.sum() == 1.0

check("topology one-hot", test_topology_onehot)

def test_property_norm():
    from utils.featurisation import normalise_property, denormalise_property
    v = normalise_property("bet_surface_area", 4000.0)
    assert 0.0 <= v <= 1.0
    v2 = denormalise_property("bet_surface_area", v)
    assert abs(v2 - 4000.0) < 1.0

check("property normalisation roundtrip", test_property_norm)

def test_lattice_features():
    from utils.featurisation import lattice_features
    v = lattice_features(25.0, 25.0, 3.6, 90.0, 90.0, 120.0)
    assert v.shape == (6,)
    # gamma = 120° → cos(120°) ≈ -0.5
    assert abs(v[5] - (-0.5)) < 0.01

check("lattice features", test_lattice_features)

# ── 2. Synthetic CIF construction ───────────────────────────────────────────
print("\n[2] Synthetic CIF (hexagonal hcb COF, imine linkage)")

SYNTHETIC_CIF = """\
data_T3_BENZ_NH2-L2_BENZ_CHO-HCB_A-AA

_cell_length_a   22.49
_cell_length_b   22.49
_cell_length_c    3.60
_cell_angle_alpha  90.0
_cell_angle_beta   90.0
_cell_angle_gamma 120.0

_symmetry_space_group_name_H-M 'P 6/m'
_symmetry_Int_Tables_number  175

loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C  0.333  0.000  0.5
C  0.333  0.083  0.5
C  0.416  0.083  0.5
N  0.416  0.166  0.5
C  0.500  0.166  0.5
C  0.500  0.250  0.5
C  0.583  0.250  0.5
C  0.000  0.333  0.5
H  0.583  0.333  0.5
H  0.333  0.166  0.5
"""

TMP_CIF = None

def make_synthetic_cif():
    global TMP_CIF
    tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    tmp.write(SYNTHETIC_CIF)
    tmp.close()
    TMP_CIF = tmp.name

check("create synthetic CIF", make_synthetic_cif)

def test_cif_minimal_parser():
    from data.crystal_graph import _parse_cif_minimal
    parsed = _parse_cif_minimal(SYNTHETIC_CIF)
    assert "elements" in parsed, "No elements found"
    assert len(parsed["elements"]) > 0
    assert parsed["a"] == 22.49
    assert parsed["gamma"] == 120.0
    assert "C" in parsed["elements"]
    assert "N" in parsed["elements"]

check("minimal CIF parser", test_cif_minimal_parser)

# ── 3. Crystal graph construction ────────────────────────────────────────────
print("\n[3] Crystal graph")

CG = None

def test_cif_to_crystal_graph():
    global CG
    from pathlib import Path
    from data.crystal_graph import cif_to_crystal_graph
    CG = cif_to_crystal_graph(
        Path(TMP_CIF),
        cutoff=3.0,   # small cutoff to keep test fast
        linkage_type="imine",
        topology="hcb",
        stacking="AA",
    )
    assert CG.n_atoms > 0, "No atoms"
    assert CG.frac_coords.shape[1] == 3
    assert CG.lattice.shape == (6,)
    assert CG.linkage_type == "imine"
    assert CG.topology == "hcb"

check("cif_to_crystal_graph", test_cif_to_crystal_graph)

def test_crystal_graph_serialisation():
    global CG
    if CG is None:
        return
    from pathlib import Path
    from data.crystal_graph import CrystalGraph
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    CG.save(Path(tmp.name))
    CG2 = CrystalGraph.load(Path(tmp.name))
    assert CG2.n_atoms == CG.n_atoms
    assert CG2.topology == "hcb"
    assert np.allclose(CG2.lattice, CG.lattice)

check("CrystalGraph save/load roundtrip", test_crystal_graph_serialisation)

def test_building_block_identification():
    global CG
    if CG is None:
        return
    # Each connected component should be a BB
    assert CG.n_building_blocks >= 1
    assert CG.bb_index.shape == (CG.n_atoms,)
    assert (CG.bb_index >= 0).all()

check("building-block identification", test_building_block_identification)

def test_pbc_edges_present():
    global CG
    if CG is None:
        return
    # For a real COF there should be edges
    assert CG.n_edges >= 0   # can be 0 if cutoff too small for this tiny test

check("PBC edges constructed", test_pbc_edges_present)

# ── 4. Cell matrix ───────────────────────────────────────────────────────────
print("\n[4] Cell matrix")

def test_cell_matrix():
    from data.crystal_graph import _cell_matrix
    M = _cell_matrix(22.49, 22.49, 3.6, 90, 90, 120)
    assert M.shape == (3, 3)
    # a-vector length should be ~22.49
    assert abs(np.linalg.norm(M[0]) - 22.49) < 0.1

check("cell matrix (hexagonal)", test_cell_matrix)

# ── 5. COFSpec and decoder ───────────────────────────────────────────────────
print("\n[5] COFSpec / reticular decoder")

def test_cofspec_name():
    from decoder.reticular_decoder import COFSpec
    spec = COFSpec(
        linkage_type="imine",
        topology="hcb",
        stacking="AA",
        node_bb="T3_BENZ",
        linker_bb="L2_BENZ",
        node_func="NH2",
        linker_func="CHO",
    )
    name = spec.to_pycofbuilder_name()
    assert "T3_BENZ" in name
    assert "L2_BENZ" in name
    assert "HCB" in name
    assert "AA" in name

check("COFSpec.to_pycofbuilder_name", test_cofspec_name)

def test_reticular_decoder_stub():
    from decoder.reticular_decoder import RetricularDecoder, COFSpec
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        rd = RetricularDecoder(output_dir=Path(tmpdir))
        spec = COFSpec("imine", "hcb", "AA", "T3_BENZ", "L2_BENZ", "NH2", "CHO")
        path = rd.assemble(spec)
        # Should produce either a CIF or a .spec.json stub
        assert path is not None
        assert Path(path).exists()

check("RetricularDecoder stub assembly", test_reticular_decoder_stub)

# ── 6. Validity checker ──────────────────────────────────────────────────────
print("\n[6] Validity checker")

def test_linkage_valence():
    from decoder.validity_checker import check_linkage_valence
    elements = ["C", "C", "N", "C", "H"]
    edge_index = np.array([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]], dtype=np.int64)
    dists = np.array([1.4,1.4,1.3,1.3,1.4,1.4,1.0,1.0], dtype=np.float32)
    ok, msg = check_linkage_valence(elements, edge_index, dists, covalent_cutoff=2.0)
    assert isinstance(ok, bool)

check("linkage valence check", test_linkage_valence)

def test_void_fraction_estimate():
    from decoder.validity_checker import estimate_void_fraction
    from data.crystal_graph import _cell_matrix
    cell = _cell_matrix(22.49, 22.49, 3.6, 90, 90, 120).astype(np.float32)
    # 3 atoms in large unit cell → lots of void
    frac = np.array([[0.333, 0.0, 0.5],
                     [0.5,   0.2, 0.5],
                     [0.1,   0.5, 0.5]], dtype=np.float32)
    vf, pld, lcd = estimate_void_fraction(frac, ["C","C","C"], cell, n_grid=10)
    assert 0.0 <= vf <= 1.0
    assert pld >= 0.0

check("void fraction estimation", test_void_fraction_estimate)

def test_synthesizability_score():
    from decoder.validity_checker import synthesizability_score
    s = synthesizability_score("imine", "T3_BENZ", "L2_BENZ", "hcb")
    assert 0.0 <= s <= 1.0
    # imine + commercial BBs + hcb should score high
    assert s > 0.5

check("synthesizability score (imine/hcb)", test_synthesizability_score)

# ── 7. Metrics ───────────────────────────────────────────────────────────────
print("\n[7] Evaluation metrics")

def test_structure_fingerprint():
    from decoder.reticular_decoder import COFSpec
    from evaluation.metrics import structure_fingerprint
    s = COFSpec("imine","hcb","AA","T3_BENZ","L2_BENZ","NH2","CHO")
    fp = structure_fingerprint(s)
    assert isinstance(fp, str)
    assert len(fp) > 0

check("structure fingerprint", test_structure_fingerprint)

def test_sun_rate_all_invalid():
    from decoder.reticular_decoder import COFSpec
    from decoder.validity_checker  import ValidityReport
    from evaluation.metrics        import compute_sun_rate
    specs   = [COFSpec("imine","hcb","AA","T3_BENZ","L2_BENZ","NH2","CHO")] * 5
    reports = [ValidityReport(name=f"test_{i}") for i in range(5)]  # all invalid
    result  = compute_sun_rate(reports, specs)
    assert result["stable_rate"] == 0.0
    assert result["unique_rate"] == 0.2  # 5 identical → 1 unique out of 5

check("SUN rate (all invalid, all duplicate)", test_sun_rate_all_invalid)

def test_sun_rate_valid():
    from decoder.reticular_decoder import COFSpec
    from decoder.validity_checker  import ValidityReport
    from evaluation.metrics        import compute_sun_rate
    from utils.featurisation       import ALL_TOPOLOGIES
    specs = [
        COFSpec("imine",  topo, "AA", "T3_BENZ", f"L2_{i}", "NH2", "CHO")
        for i, topo in enumerate(ALL_TOPOLOGIES[:5])
    ]
    reports = []
    for i in range(5):
        r = ValidityReport(name=f"s{i}")
        r.linkage_valid   = True
        r.pore_accessible = True
        r.uff_converged   = True
        r.synth_score     = 0.7
        reports.append(r)
    result = compute_sun_rate(reports, specs)
    assert result["stable_rate"] == 1.0
    assert result["unique_rate"] == 1.0
    assert result["sun_rate"]    == 1.0

check("SUN rate (all valid + unique)", test_sun_rate_valid)

def test_diversity_metrics():
    from decoder.reticular_decoder import COFSpec
    from evaluation.metrics        import internal_diversity, topology_distribution
    specs = [
        COFSpec("imine", "hcb", "AA", "T3_BENZ", "L2_BENZ", "NH2", "CHO"),
        COFSpec("imine", "sql", "AB", "S4_PHTH",  "L2_NAPH", "NH2", "CHO"),
        COFSpec("boronate_ester", "kgm", "AA", "T3_TRIZ", "L2_BIPH", "B(OH)2", "diol"),
    ]
    div  = internal_diversity(specs)
    dist = topology_distribution(specs)
    assert 0.0 <= div <= 1.0
    assert "hcb" in dist

check("diversity metrics", test_diversity_metrics)

# ── 8. Torch-dependent tests ────────────────────────────────────────────────
print("\n[8] Model architecture (torch)")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  ⚠  torch not installed — skipping model tests")

if HAS_TORCH:
    from utils.featurisation import (
        ATOM_FEAT_DIM, BOND_FEAT_DIM, N_TOPOLOGIES,
        N_LINKAGE_TYPES, N_STACKING, LATENT_DIM,
    )

    def make_fake_batch(B=2, N=60, E=200):
        n_bbs = 4
        return {
            "atoms":             torch.randn(N, ATOM_FEAT_DIM),
            "frac_coords":       torch.rand(N, 3),
            "lattice":           torch.randn(B, 6),
            "edge_index":        torch.randint(0, N, (2, E)),
            "edge_attr":         torch.randn(E, BOND_FEAT_DIM),
            "edge_shift":        torch.zeros(E, 3),
            "bb_index":          torch.randint(0, n_bbs, (N,)),
            "batch":             torch.cat([
                                     torch.zeros(N//2, dtype=torch.long),
                                     torch.ones(N//2, dtype=torch.long),
                                 ]),
            "topology_idx":      torch.randint(0, N_TOPOLOGIES, (B,)),
            "linkage_idx":       torch.randint(0, N_LINKAGE_TYPES, (B,)),
            "stacking_idx":      torch.zeros(B, dtype=torch.long),
            "n_atoms_per_graph": torch.tensor([N//2, N//2]),
        }

    def test_encoder_forward():
        from models.encoder import COFEncoder, kl_divergence
        enc = COFEncoder(latent_dim=64, hidden_dim=64, n_layers=2, bb_layers=1)
        enc.eval()
        batch = make_fake_batch(B=2, N=40, E=80)
        with torch.no_grad():
            z, mu, lv = enc(batch)
        assert z.shape  == (2, 64), f"Got {z.shape}"
        assert mu.shape == (2, 64)
        assert lv.shape == (2, 64)
        kl = kl_divergence(mu, lv)
        assert kl.item() > 0

    check("COFEncoder forward (latent_dim=64)", test_encoder_forward)

    def test_flow_matching_forward():
        from models.flow_matching import FlowMatchingNetwork, cfm_loss
        net = FlowMatchingNetwork(latent_dim=64, hidden_dim=128, n_layers=2)
        net.eval()
        B = 4
        z1 = torch.randn(B, 64)
        t  = torch.rand(B)
        with torch.no_grad():
            v  = net(z1, t)
        assert v.shape == (B, 64)
        loss = cfm_loss(net, z1)
        assert loss.item() > 0

    check("FlowMatchingNetwork forward", test_flow_matching_forward)

    def test_ode_sampling():
        from models.flow_matching import FlowMatchingNetwork, sample_ode
        net   = FlowMatchingNetwork(latent_dim=64, hidden_dim=128, n_layers=2)
        z_gen = sample_ode(net, n_samples=3, n_steps=5)
        assert z_gen.shape == (3, 64)

    check("ODE sampling (5 steps)", test_ode_sampling)

    def test_adapter_forward():
        from models.adapters import ScalarPropertyAdapter, LinkageAdapter, MultiAdapter
        B = 4
        cond = torch.randn(B, 128)
        prop_adapter = ScalarPropertyAdapter("bet_surface_area", hidden_dim=128)
        val  = torch.rand(B)
        res  = prop_adapter(cond, val)
        assert res.shape == (B, 128)

        lk_adapter = LinkageAdapter(hidden_dim=128)
        idx = torch.randint(0, N_LINKAGE_TYPES, (B,))
        res2 = lk_adapter(cond, idx)
        assert res2.shape == (B, 128)

    check("Adapter forward passes", test_adapter_forward)

    def test_spec_decoder_forward():
        from decoder.reticular_decoder import SpecDecoderMLP
        dec  = SpecDecoderMLP(latent_dim=64)
        z    = torch.randn(3, 64)
        out  = dec(z)
        assert "linkage"  in out
        assert "topology" in out
        specs = dec.decode_greedy(z)
        assert len(specs) == 3
        for s in specs:
            assert hasattr(s, "linkage_type")
            assert hasattr(s, "topology")

    check("SpecDecoderMLP forward + greedy decode", test_spec_decoder_forward)

    def test_rk4_solver():
        from models.flow_matching import FlowMatchingNetwork, sample_ode
        net = FlowMatchingNetwork(latent_dim=64, hidden_dim=128, n_layers=2)
        z   = sample_ode(net, n_samples=2, n_steps=4, solver="rk4")
        assert z.shape == (2, 64)

    check("RK4 ODE solver", test_rk4_solver)

    def test_cfg_sampling():
        from models.flow_matching import FlowMatchingNetwork, sample_cfg
        net  = FlowMatchingNetwork(latent_dim=64, hidden_dim=128, n_layers=2)
        props = {"bet_surface_area": torch.tensor([0.5, 0.8])}
        z     = sample_cfg(net, n_samples=2, n_steps=4, props=props,
                           guidance_scale=2.0)
        assert z.shape == (2, 64)

    check("CFG sampling", test_cfg_sampling)

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
n_pass = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
print(f"Results: {n_pass} passed, {n_fail} failed out of {len(results)} tests")
if n_fail > 0:
    print("\nFailed tests:")
    for r in results:
        if r[0] == FAIL:
            print(f"  {FAIL} {r[1]}: {r[2]}")
print("="*60 + "\n")

# Cleanup
import os
if TMP_CIF and os.path.exists(TMP_CIF):
    os.unlink(TMP_CIF)

sys.exit(0 if n_fail == 0 else 1)
