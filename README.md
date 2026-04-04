# COFGen — COF Property Prediction & Design Tool

**Predict, simulate, and reverse-engineer Covalent Organic Frameworks from building blocks alone.**

COFGen computes pore geometry, simulates powder X-ray diffraction, predicts stacking preferences, estimates gas adsorption, band gap, mechanical properties, stability, and generates synthesis conditions — in seconds, no DFT required.

---

## Quickstart

```bash
git clone https://github.com/your-username/cofgen.git
cd cofgen
python3 -m venv cofgen_env && source cofgen_env/bin/activate
pip install -r requirements.txt
python3 cofgen_tool.py predict --node T3_BENZ --linker L2_PYRN --linkage imine
```

---

## Commands

```bash
# Full property prediction from BB names
python3 cofgen_tool.py predict --node T3_BENZ --linker L2_PYRN --linkage imine

# From a CIF file
python3 cofgen_tool.py predict --cif my_structure.cif --json results.json

# Simulate PXRD and save plot
python3 cofgen_tool.py pxrd --node T3_BENZ --linker L2_PYRN --linkage imine --plot

# Stacking geometry analysis
python3 cofgen_tool.py stacking --cif my_structure.cif

# Synthesis conditions + DoE matrix
python3 cofgen_tool.py synthesis --node T3_BENZ --linker L2_PYRN --linkage imine

# Diagnose a failed synthesis
python3 cofgen_tool.py synthesis --node T3_BENZ --linker L2_PYRN --linkage imine \
    --observation "gel-like precipitate, no filterable powder"

# Reverse-engineer building blocks from a CIF
python3 cofgen_tool.py reverse --cif unknown_cof.cif

# Generate candidates targeting properties
python3 cofgen_tool.py generate --linkage imine --co2 3.5 --bet 2000 -n 30

# List all building blocks
python3 cofgen_tool.py list-bbs
```

Add `--json output.json` to any command to save structured results.

---

## What gets predicted

| Property | Output |
|---|---|
| Pore geometry | Void fraction, BET surface area, PLD, LCD, density, cell parameter |
| PXRD | Full simulated pattern with (hkl) peak list (Cu Kα, Cromer-Mann scattering factors) |
| Stacking | AA/AB/ABC probability, layer spacing, lateral offset |
| Gas adsorption | CO₂ @ 298K/1bar · CH₄ @ 298K/65bar · H₂ @ 77K/100bar · N₂ @ 298K/1bar · CO₂/N₂ selectivity |
| Electronics | Band gap (eV) with semiconductor classification |
| Mechanical | In-plane and out-of-plane Young's modulus, bulk modulus, interlayer shear |
| Stability | Thermal decomposition temp, water/acid/base stability, hydrolysis half-life |
| Synthesis | Solvent, catalyst, temperature, time, 10-experiment DoE matrix |
| Failure diagnosis | Class A–D per Chen et al. JACS 2026 taxonomy |

---

## Building blocks

### Tritopic nodes (hcb, kgm, hxl topologies)
`T3_BENZ` · `T3_TRIZ` · `T3_TPM` · `T3_TPA` · `T3_TRIF` · `T3_INTZ`

### Tetratopic nodes (sql topology)
`S4_BENZ` · `S4_PORPH` · `S4_PHTH`

### Linkers
`L2_BENZ` · `L2_NAPH` · `L2_BIPH` · `L2_TPHN` · `L2_ANTR` · `L2_PYRN` · `L2_AZBN` · `L2_ETBE` · `L2_STIL` · `L2_BTTA`

### Linkage types
`imine` · `boronate_ester` · `boroxine` · `beta_ketoenamine` · `hydrazone` · `triazine`

---

## Repository structure

```
cofgen/
├── cofgen_tool.py                    ← CLI entry point
├── requirements.txt
├── analysis/
│   ├── pxrd_simulator.py             ← structure factor PXRD + stacking
│   ├── property_predictor.py         ← band gap, adsorption, mechanical, stability
│   └── monomer_reverse_engineer.py   ← CIF → building block identity + SMILES
├── data/
│   ├── crystal_graph.py              ← CIF parser → CrystalGraph
│   ├── property_labels.py            ← geometric BET, void fraction, PLD
│   └── synthetic_cif_generator.py    ← reticular geometry CIF builder
├── models/
│   ├── synthesis_condition_predictor.py  ← RAG over CS-KB (Chen et al. 2026)
│   ├── synthesizability.py           ← GBDT synthesizability classifier
│   ├── encoder.py                    ← graph transformer VAE (torch, optional)
│   ├── flow_matching.py              ← flow matching generator (torch, optional)
│   └── mattersim_stability.py        ← MatterSim interface + GA/SmVAE baselines
├── decoder/
│   ├── reticular_decoder.py          ← BB library, COFSpec
│   └── validity_checker.py           ← pore + synth validity
├── scripts/                          ← training pipeline (GPU)
├── evaluation/                       ← SUN rate, diversity, benchmark
├── utils/featurisation.py
└── tests/smoke_test.py               ← 21 tests, no GPU required
```

---

## Training the generative model (GPU required)

```bash
# Get data
# ReDD-COFFEE → data/raw/redd_coffee/   https://doi.org/10.1039/D3TA00470H
# CoRE-COF   → data/raw/core_cof/       https://core-cof.github.io/

pip install torch torch-geometric pymatgen pyCOFBuilder rdkit-pypi

python scripts/build_dataset.py  --raw_dir data/raw/ --out_dir data/processed/ --n_jobs 32
python scripts/compute_properties.py --data data/processed/ --n_jobs 32
python scripts/train_encoder.py      --data data/processed/ --epochs 100 --device cuda
python scripts/train_flowmatch.py    --encoder_ckpt checkpoints/encoder/best.pt
python scripts/train_adapter.py      --property co2_uptake_298k_1bar

python scripts/generate.py \
  --checkpoint checkpoints/flowmatch_base/best.pt \
  --adapter_co2 checkpoints/adapter_co2/best.pt \
  --co2_target 4.0 --bet_target 2500 --linkage imine --n 500
```

---

## Citation

```bibtex
@software{cofgen2026,
  title = {COFGen: Property Prediction and Generative Design for Covalent Organic Frameworks},
  year  = {2026},
  url   = {https://github.com/jishnucv/cofgen}
}
```

Synthesis conditions use the workflow from:

> Chen, L. et al. *Chemist-Guided Human–AI Workflow for Covalent Organic Framework Synthesis.* J. Am. Chem. Soc. **148**, 7440–7449 (2026). https://doi.org/10.1021/jacs.5c20068

---

## License

MIT — see [LICENSE](LICENSE)
