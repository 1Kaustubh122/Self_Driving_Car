import zipfile

try:
    with zipfile.ZipFile("checkpoints/sac_carla_model_117200_steps.zip", 'r') as zf:
        print("Valid zip contents:", zf.namelist())
except zipfile.BadZipFile:
    print("Not a valid zip file!")
