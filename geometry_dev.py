import lsdo_geo as lsdo_geo
import numpy as np
import csdl_alpha as csdl
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
import lsdo_function_spaces as lfs
lfs.num_workers = 1

from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block, construct_ffd_block_around_entities, construct_ffd_block_from_corners
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)


recorder = csdl.Recorder(inline=True)
recorder.start()

geometry = lsdo_geo.import_geometry('bwbv2nc.stp')
# geometry.plot(show=True)



# manually selected indices for the different surfaces
_left_wing_indices = [16,17,18,19,20,21,22,23]
_center_wing_indices = [12,13,14,15,0,1,2,3]
_right_wing_indices = [4,5,6,7,8,9,10,11]


left_wing = geometry.declare_component(_left_wing_indices)
# left_wing.plot()

center_wing = geometry.declare_component(_center_wing_indices)
# center_wing.plot()

right_wing = geometry.declare_component(_right_wing_indices)
# right_wing.plot()



# num_ffd_sections = 5
# num_wing_secctions = 3
# ffd_block = construct_ffd_block_around_entities(entities=geometry, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))
# ffd_block = construct_tight_fit_ffd_block(entities=geometry, num_coefficients=(2, (num_ffd_sections // num_wing_secctions + 1), 2), degree=(1,1,1))

# center wing ffd block
center_wing_ffd_block = construct_ffd_block_around_entities(entities=center_wing, num_coefficients=(10,3,2), degree=(1,1,1))
# center_wing_ffd_block.plot()

# left wing ffd block
# left_wing_ffd_block = construct_ffd_block_around_entities(entities=left_wing, num_coefficients=(2,2,2), degree=(1,1,1))
left_wing_ffd_block = construct_ffd_block_around_entities(entities=left_wing, num_coefficients=(2,2,2), degree=(1,1,1))
# left_wing_ffd_block.plot()

# right wing ffd block
right_wing_ffd_block = construct_ffd_block_around_entities(entities=right_wing, num_coefficients=(2,2,2), degree=(1,1,1))
# right_wing_ffd_block.plot()




left_leading_edge = geometry.project(np.array([29.019, -25.852, 2.123]), plot=False)
right_leading_edge = geometry.project(np.array([29.019, 25.852, 2.123]), plot=False)

left_center_section_leading_edge = geometry.project(np.array([17.815, -9.891, 1.04]), plot=False)
right_center_section_leading_edge = geometry.project(np.array([17.815, 9.891, 1.04]), plot=False)

dog_tail_1 = geometry.project(np.array([30.0, 0.0, 0.0]), plot=False)
dog_tail_2 = geometry.project(np.array([27.148, 0.0, 0.476]), plot=False)







# i don't know what this does tbh...
center_wing_ffd_sectional_parameterization = VolumeSectionalParameterization(name="center_wing_ffd_sectional_parameterization", 
                                                                             parameterized_points=center_wing_ffd_block.coefficients, 
                                                                             principal_parametric_dimension=0,)
# center_wing_ffd_sectional_parameterization.plot()

left_wing_ffd_sectional_parameterization = VolumeSectionalParameterization(name="left_wing_ffd_sectional_parameterization", 
                                                                           parameterized_points=left_wing_ffd_block.coefficients, 
                                                                           principal_parametric_dimension=1,)
# left_wing_ffd_sectional_parameterization.plot()

right_wing_ffd_sectional_parameterization = VolumeSectionalParameterization(name="right_wing_ffd_sectional_parameterization", 
                                                                           parameterized_points=right_wing_ffd_block.coefficients, 
                                                                           principal_parametric_dimension=1,)
# right_wing_ffd_sectional_parameterization.plot()


# i don't really know what this does either...
space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))











# declare design variables
# wing_span_dv = csdl.Variable(value=58.038)
wing_span_dv = csdl.Variable(value=98.038)
wing_span_array = csdl.Variable(value=np.zeros(2))
wing_span_array = wing_span_array.set(csdl.slice[0], -10)
wing_span_dv.set_as_design_variable(lower=1, scaler=1E-1)

center_span_dv = csdl.Variable(value=19.782)
center_span_dv.set_as_design_variable(lower=1, scaler=1E-1)



# define wingspan as a function of the projected left and right leading edges
wing_span = csdl.norm(geometry.evaluate(left_leading_edge) - geometry.evaluate(right_leading_edge))
# print('wing span: ', wing_span.value)

# define center span as a function of the projected left and right center section leading edges
center_span = csdl.norm(geometry.evaluate(left_center_section_leading_edge) - geometry.evaluate(right_center_section_leading_edge))
# print('center span: ', center_span.value)







# parametric_b_spline_inputs = np.linspace(0.0, 1.0, 10).reshape((-1, 1))


# wing_span_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
#                                             coefficients=csdl.ImplicitVariable(shape=(2,), value=np.array([0., 0.])), 
#                                             name='wingspan_stretching_b_spline_coefficients')

# wingspan_stretch_sectional_parameters = wing_span_stretching_b_spline.evaluate(parametric_b_spline_inputs)


# don't know what this does....
wing_span_sectional_parameters = VolumeSectionalParameterizationInputs()
wing_span_sectional_parameters.add_sectional_translation(axis=1, translation=wing_span_array)

wing_span_ffd_coefficients = left_wing_ffd_sectional_parameterization.evaluate(wing_span_sectional_parameters, plot=True)
left_wing_ffd_block.coefficients = wing_span_ffd_coefficients
left_wing.set_coefficients(left_wing_ffd_block.evaluate(wing_span_ffd_coefficients, plot=False))



geometry.plot()


exit()

# do everything with inner optimization ?????

geometry_solver = ParameterizationSolver()
geometry_solver.add_parameter(wing_span_stretching_b_spline.coefficients)

geometric_variables = GeometricVariables()
geometric_variables.add_variable(wing_span, wing_span_dv)
geometric_variables.add_variable(center_span, center_span_dv)



# geometry.plot()
geometry_solver.evaluate(geometric_variables)
geometry.plot()
center_wing_ffd_block.plot()


exit()





"""
# note: make these zero to not modify initial geom during fwd run
chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                         coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0.,  0.])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0.,  0.])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0.,  0.])*np.pi/180), name='twist_b_spline_coefficients')




parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
    parametric_b_spline_inputs
)

twist_sectional_parameters = twist_b_spline.evaluate(
    parametric_b_spline_inputs
)



sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)




geometry_coefficients = center_wing_ffd_block.evaluate(ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients)
# geometry.plot()




wingspan = csdl.norm(
    geometry.evaluate(leading_edge_right) - geometry.evaluate(leading_edge_left)
)
# root_chord = csdl.norm(
#     geometry.evaluate(trailing_edge_center) - geometry.evaluate(leading_edge_center)
# )
# tip_chord_left = csdl.norm(
#     geometry.evaluate(trailing_edge_left) - geometry.evaluate(leading_edge_left)
# )
# tip_chord_right = csdl.norm(
#     geometry.evaluate(trailing_edge_right) - geometry.evaluate(leading_edge_right)
# )


# spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
# spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
# # sweep_angle = csdl.arccos(csdl.vdot(spanwise_direction, np.array([0., -1., 0.])) / csdl.norm(spanwise_direction))
# sweep_angle_left = csdl.arccos(-spanwise_direction_left[1] / csdl.norm(spanwise_direction_left))
# sweep_angle_right = csdl.arccos(spanwise_direction_right[1] / csdl.norm(spanwise_direction_right))




wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([75]))
# root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([2.0]))
# tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.5]))
# sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([45*np.pi/180]))

"""

num_sections=10
elevator_rot = csdl.Variable(value=np.deg2rad(45))
rot_array = csdl.Variable(value=np.zeros(num_sections))
rot_array = rot_array.set(csdl.slice[num_sections - 1], -1 * elevator_rot)

elevator_z_trans = csdl.Variable(value=0)
trans_z_array = csdl.Variable(value=np.zeros(num_sections))
trans_z_array = trans_z_array.set(csdl.slice[num_sections - 1], elevator_z_trans)

elevator_x_trans = csdl.Variable(value=0)
trans_x_array = csdl.Variable(value=np.zeros(num_sections))
trans_x_array = trans_x_array.set(csdl.slice[num_sections - 1], elevator_x_trans)


elevator_length = csdl.Variable(value=0.3)


sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_translation(axis=0, translation=trans_x_array)
sectional_parameters.add_sectional_translation(axis=2, translation=trans_z_array)
sectional_parameters.add_sectional_rotation(axis=1, rotation=rot_array)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

center_wing_ffd_block.coefficients = ffd_coefficients

center_wing.set_coefficients(center_wing_ffd_block.evaluate(ffd_coefficients, plot=False))
# center_wing.plot()
# exit()

vec = center_wing.evaluate(dog_tail_1) - center_wing.evaluate(dog_tail_2)
x, z = vec[0], vec[2]
actual_rot = csdl.arctan(z / x)
fake_length = csdl.norm(vec)


geometry_solver = ParameterizationSolver()
geometry_solver.add_parameter(elevator_z_trans)
geometry_solver.add_parameter(elevator_x_trans)

geometric_variables = GeometricVariables()
geometric_variables.add_variable(actual_rot, elevator_rot)
geometric_variables.add_variable(fake_length, elevator_length)

# geometry.plot()
geometry_solver.evaluate(geometric_variables)
geometry.plot()
center_wing_ffd_block.plot()





recorder.stop()