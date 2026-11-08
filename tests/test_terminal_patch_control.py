from crimm.Modeller.SeqChainGenerator import SeqChainGenerator
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Model import Model
from crimm.StructEntities.OrganizedModel import OrganizedModel
from crimm.StructEntities.Residue import Residue


def _protein_chain(amidated):
    generator = SeqChainGenerator()
    generator.set_sequence("AAA", "protein")
    chain = generator.create_chain()
    if amidated:
        cap = Residue(("H_NH2", 4, " "), "NH2", " ")
        cap.add(Atom("N", chain[3]["C"].coord + [1.3, 0, 0], element="N"))
        chain.add(cap)
    return chain


def test_explicit_none_disables_c_terminal_patch():
    for amidated in (False, True):
        for model_level in (False, True):
            chain = _protein_chain(amidated)
            topology = TopologyGenerator()
            if model_level:
                model = Model(0)
                model.add(chain)
                model = OrganizedModel(model, fetch_web_data=False)
                topology.generate_model(model, prot_last_patch=None, QUIET=True)
                chain = model.protein[0]
            else:
                topology.generate(chain, first_patch="ACE", last_patch=None, QUIET=True)
            terminal = chain.residues[-1]
            assert terminal.topo_definition.patch_with is None
            assert "NT" not in terminal.topo_definition


def test_default_c_terminal_patch_preserves_input_amidation():
    for amidated in (False, True):
        chain = _protein_chain(amidated)
        model = Model(0)
        model.add(chain)
        model = OrganizedModel(model, fetch_web_data=False)
        TopologyGenerator().generate_model(model, QUIET=True)
        terminal = model.protein[0].residues[-1]
        assert terminal.topo_definition.patch_with == ("CT2" if amidated else "CT3")
        assert "NT" in terminal
        assert ("CAT" in terminal) == (not amidated)
