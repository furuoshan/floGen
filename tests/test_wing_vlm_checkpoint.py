"""VLM provenance must not masquerade as a trained SLD network."""
import unittest
import main as m

class VLMCheckpointTests(unittest.TestCase):
    def test_vlm_accepts_only_geometry_only_external_sld(self):
        info={'airfoil_training_steps':10,'sld_training_steps':0,'sld_network_used':False,
              'sld_source':'pytornado_vlm','sld_uses_cfd':False}
        m.validate_wing_priors(info,external_sld=True)
        with self.assertRaisesRegex(ValueError,'predict-foilpara'):
            m.validate_wing_priors(info)
        for changes in ({'sld_uses_cfd':True},{'sld_source':'cfd_integral'},{'airfoil_training_steps':0}):
            with self.assertRaises(ValueError):
                m.validate_wing_priors(dict(info,**changes),external_sld=True)

    def test_trained_sld_predecessor_still_required_for_old_mode(self):
        m.validate_wing_priors({'airfoil_training_steps':10,'sld_training_steps':1})
        with self.assertRaises(ValueError):
            m.validate_wing_priors({'airfoil_training_steps':10,'sld_training_steps':0},external_sld=True)

if __name__=='__main__':unittest.main()
