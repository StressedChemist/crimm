import os, tarfile, shutil

md_input = [
'toppar.str',
'step3_size.str',
'step5_assembly.str',
'step3_nlipids_upper.prm',
'step3_nlipids_lower.prm',
'step3_lipid_conjugate.str',
'step5_assembly.psf',
'step5_assembly.hmr.psf',
'step5_assembly.crd',
'crystal_image.str',
'checkfft.py',
'checkfft.str',
'membrane_restraint.str',
'membrane_restraint2.str',
'carbohydrate_restraint.str',
'step6.1_equilibration.inp',
'step6.2_equilibration.inp',
'step6.3_equilibration.inp',
'step6.4_equilibration.inp',
'step6.5_equilibration.inp',
'step6.6_equilibration.inp',
'step7_production.inp',
# directory
'toppar',
'openmm',
'gromacs',
'amber',
]

jobid = os.path.basename(os.getcwd())
folder = f'md_input_{jobid}'
if not os.path.exists(folder): os.mkdir(folder)
has_input = False
for name in md_input:
    if os.path.exists(name):
        has_input = True
        if os.path.isdir(name):
            shutil.copytree(name, folder+'/'+name)
        else:
            print(name)
            print(folder+'/'+name)
            shutil.copy(name, folder+'/'+name)
if has_input:
    tar = tarfile.open('md_input.tgz', 'w:gz')
    tar.add(folder)
    tar.close()
