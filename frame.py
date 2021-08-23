import numpy as np

class Frame:
	
	def __init__(self, inputs_for_networks, energy_of_frame):
		self._inputs_for_networks = inputs_for_networks
		self._energy_of_frame = energy_of_frame
	
	def get_input(self,atom_type):
		# note that atom_type for now has to be a number 0,1,...,n-1.
		# later, there should exist a global dictionary somewhere mapping
		# the atomic symbol or the atomic number to 0,...,n-1, so that
		# all the lists are sorted in the same way
		return self._inputs_for_networks[atom_type]
	
	def get_inputs(self):
		return self._inputs_for_networks

	def get_energy_of_frame(self):
		return self._energy_of_frame

	def get_energy_of_frame_as_array(self):
		return np.array([self._energy_of_frame])

	def get_energy_of_frame_as_matrix(self):
		return np.array([[self._energy_of_frame]])

	def shift_and_scale(self,shift_values_input,scale_values_input,shift_value_output,scale_value_output):
		# shift values and scale values shall be a list of vectors,
		# one vector for each atom type with length according to the
		# number of features for this atom type
		for i in range(len(self._inputs_for_networks)):
			self._inputs_for_networks[i] -= shift_values_input[i]
			self._inputs_for_networks[i] = np.divide(self._inputs_for_networks[i],scale_values_input[i])
		self._energy_of_frame -= shift_value_output
		self._energy_of_frame = np.divide(self._energy_of_frame,scale_value_output)
