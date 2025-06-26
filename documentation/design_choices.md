# Design choices



# Pyproject.toml 

I have looked at discussion about ros2 compatibility. https://ros2-tutorial.readthedocs.io/en/latest/preamble/python/python_packaging.html . 
However, it feels like this is the new way. In addition, this library is not aimed at becoming a ros2 node but a requirements of a ros2 package. Thus, it should be alright. 

## Test environnement

The strategy used to dev the test is *Tests outside application code* . Thus, the test are in 
the tests folder with a structure mirrorring the structure of the src/ folder that contains 
the main code. 

To test the code. Simply run pytest in the commandline. 



## __init__.py

Theser folder contains lines to make the import works. 

The structures : 
    from . import kinematics
    from . import dynamics

Allows people to import norlab_motion_models.motion_models for example and then use motion_models.kinematic.ideal_diff_drive.add_one(1). 

