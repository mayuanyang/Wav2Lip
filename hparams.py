from glob import glob
import os

def get_image_list(data_root, split, train_root='filelists'):
	filelist = []

	with open('{}/{}.txt'.format(train_root, split)) as f:
		for line in f:
			line = line.strip()
			if ' ' in line: line = line.split()[0]
			filelist.append(os.path.join(data_root, line))

	return filelist

class HParams:
	def __init__(self, **kwargs):
		self.data = {}

		for key, value in kwargs.items():
			self.data[key] = value

	def __getattr__(self, key):
		if key not in self.data:
			raise AttributeError("'HParams' object has no attribute %s" % key)
		return self.data[key]

	def set_hparam(self, key, value):
		self.data[key] = value


# Default hyperparameters
hparams = HParams(
	num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
	#  network
	rescale=True,  # Whether to rescale audio prior to preprocessing
	rescaling_max=0.9,  # Rescaling value
	
	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	
	n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
	hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
	win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
	sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
	
	frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
	
	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization=True,
	# Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
	symmetric_mels=True,
	# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
	# faster and cleaner convergence)
	max_abs_value=4.,
	# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
	# be too big to avoid gradient explosion, 
	# not too small for fast convergence)
	# Contribution by @begeekmyfriend
	# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
	# levels. Also allows for better G&L phase reconstruction)
	preemphasize=True,  # whether to apply filter
	preemphasis=0.97,  # filter coefficient.
	
	# Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=55,
	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax=7600,  # To be increased/reduced depending on data.

	###################### Our training parameters #################################
	img_size=192,
	fps=25,
	
	batch_size=7,
	initial_learning_rate=1e-3,
	nepochs=200000000000000000,  ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
	resunet_num_workers=48,
	checkpoint_interval=1000,
	eval_interval=500000,
  save_optimizer_state=True,
	image_cache_size=455000,
	audio_cache_size=35000,

  
	syncnet_num_workers=24,
	syncnet_batch_size=530,
	syncnet_face_lr=6e-6,
	syncnet_audio_lr=8e-6,
	syncnet_eval_interval=100000,
	syncnet_checkpoint_interval=200,
	syncnet_image_cache_size=35000,
	syncnet_audio_cache_size=3000,

  syncnet_wt=0.4,
  disc_wt=0.5,
	bottom_disc_wt=0.6,
  l1_wt=0.4,
	bottom_l1_wt=0.35,
  ssim_wt=0.0,
	
  
)


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)
