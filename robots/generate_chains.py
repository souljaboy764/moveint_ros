# IKPy imports
from ikpy import chain
from ikpy.urdf.utils import get_urdf_tree

# Generate the pdf
dot, urdf_tree = get_urdf_tree("/home/ip-bihandovers/ip_bimanual/kobo_ros_core_ias/src/ip_bimanual/robots/blaise_panda_dual.urdf", out_image_path="./panda", root_element="base_link")


########################## Left arm ##########################

panda_left_arm_links = ["base_link",
                         "torso",
                         "panda_left_link0",
                         "panda_left_link1",
                         "panda_left_link2",
                         "panda_left_link3",
                         "panda_left_link4",
                         "panda_left_link5",
                         "panda_left_link6",
                         "panda_left_link7",
                         "panda_left_link8"]

panda_left_arm_joints = ["torso_joint",
                          "panda_left_joint_torso",
                          "panda_left_joint1",
                          "panda_left_joint2",
                          "panda_left_joint3",
                          "panda_left_joint4",
                          "panda_left_joint5",
                          "panda_left_joint6",
                          "panda_left_joint7",
                          "panda_left_joint8"]

panda_left_arm_elements = [x for pair in zip(panda_left_arm_links, panda_left_arm_joints) for x in pair]
# Remove the gripper, it's weird
# pepper_left_arm_elements = [x for pair in zip(pepper_left_arm_links, pepper_left_arm_joints) for x in pair][:-3]

panda_left_arm_chain = chain.Chain.from_urdf_file(
    "/home/ip-bihandovers/ip_bimanual/kobo_ros_core_ias/src/ip_bimanual/robots/blaise_panda_dual.urdf",
    base_elements=panda_left_arm_elements,
    last_link_vector=[0, 0.0, 0.1034],
    active_links_mask=3*[False] + 7 * [True] + 4*[False],
    symbolic=False,
    name="panda_left_arm")
panda_left_arm_chain.to_json_file(force=True)

############################## Right arm ##############################

panda_right_arm_elements = [x.replace("left", "right") for x in panda_left_arm_elements]

panda_right_arm_chain = chain.Chain.from_urdf_file(
    "/home/ip-bihandovers/ip_bimanual/kobo_ros_core_ias/src/ip_bimanual/robots/blaise_panda_dual.urdf",
    base_elements=panda_right_arm_elements,
    last_link_vector=[0, 0.0, 0.1034],
    active_links_mask=3*[False] + 7 * [True] + 4*[False],
    symbolic=False,
    name="panda_right_arm")
panda_right_arm_chain.to_json_file(force=True)