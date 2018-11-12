import netdef_slim as nd
import os

nd.evo_manager.set_training_dir(os.path.join(os.path.dirname(__file__), 'training'))
schedule = nd.FixedStepSchedule('S_custom', max_iter=220000, steps=[220000], base_lr=1e-04)
nd.add_evo(nd.Evolution('flyingThings3D.train', [], schedule))
