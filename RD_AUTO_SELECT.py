# %%
# Copyright (C) 2024 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# THIS WORK IS BASE ON jonkraft's WORK OUR TEAM MADE SOME MODIFICATIONS TO THE ORIGINAL CODE 
# TO IMPROVE THE PERFORMANCE AND STABILITY OF THE SYSTEM. THE ORIGINAL CODE CAN BE FOUND AT
# https://github.com/jonkraft/PhaserRadarLabs/blob/main/Range_Doppler_Plot.py
import os
import sys
import time
import json
import threading
import queue
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
import adi
import scipy.ndimage as ndimage
import matplotlib.patches as patches
import atexit


fs = 4e6
c = 3e8
center_freq = 2.1e9
f0 = 100e3
f_out = 9.9e9
BW = 500e6
ramp_time_us = 300
ramp_time_ms = ramp_time_us * 1e-3
ramp_time_s = ramp_time_us * 1e-6
max_range = 20
delay_time_ms = 0.1
PRI_ms = ramp_time_ms + delay_time_ms
PRI_s = PRI_ms * 1e-3
N_frame = int(PRI_s * float(fs))
useable_point = int(ramp_time_s * fs)
delay_point = int(delay_time_ms * 1e-3 * fs)
slope = BW / ramp_time_s
freq = np.linspace(-fs / 2, fs / 2, N_frame)
dist = (freq - f0) * c / (2 * slope)
lam = c / f_out
rx_gain = 40
range_min = 8
range_max = 30
MTI_t = 1
num_chirps = 128 + MTI_t

CLASS_NAMES = ["rock", "branch", "man", "metal"]
DISPLAY_CLASS_NAMES = ["ROCK", "BRANCH", "MAN", "METAL"]
MODEL_PATCH_SIZE = 16
MODEL_RD_PATCH_HW = (16, 16)
MODEL_IQ_PATCH_HW = (129, 16)
MODEL_INPUT_STANDARDIZE = False
MODEL_CLIP_STD = 5.0


def resolve_model_npz_path():
    env_path = os.environ.get("RADAR_MODEL_NPZ")
    if env_path and os.path.exists(env_path):
        return env_path

    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "cn0566_dual_branch_best.npz"),
        os.path.join(os.path.dirname(__file__), "cn0566_model_output", "cn0566_dual_branch_best.npz"),
        os.path.join(os.path.dirname(__file__), "best_model_2.npz"),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return candidate_paths[-1]


def load_model_metadata(npz_path):
    metadata = {
        "class_names": CLASS_NAMES[:],
        "display_class_names": DISPLAY_CLASS_NAMES[:],
        "rd_patch_hw": MODEL_RD_PATCH_HW,
        "iq_patch_hw": MODEL_IQ_PATCH_HW,
        "input_standardize": MODEL_INPUT_STANDARDIZE,
    }

    if not npz_path or not os.path.exists(npz_path):
        return metadata

    try:
        data = np.load(npz_path, allow_pickle=False)
        if "class_names_json" in data:
            class_names = json.loads(str(data["class_names_json"]))
            if isinstance(class_names, list) and class_names:
                metadata["class_names"] = [str(name) for name in class_names]
                metadata["display_class_names"] = [str(name).upper() for name in class_names]

        if "train_summary_json" in data:
            summary = json.loads(str(data["train_summary_json"]))
            if isinstance(summary, dict):
                input_shapes = summary.get("input_shapes", {})
                rd_shape = input_shapes.get("rd_patch")
                iq_shape = input_shapes.get("iq_patch")
                if isinstance(rd_shape, list) and len(rd_shape) == 3:
                    metadata["rd_patch_hw"] = (int(rd_shape[1]), int(rd_shape[2]))
                if isinstance(iq_shape, list) and len(iq_shape) == 3:
                    metadata["iq_patch_hw"] = (int(iq_shape[1]), int(iq_shape[2]))
                metadata["input_standardize"] = bool(summary.get("input_standardize", metadata["input_standardize"]))
    except Exception as exc:
        print(f"[模型] 讀取 metadata 失敗，改用內建預設: {exc}")

    return metadata


MODEL_NPZ_PATH = resolve_model_npz_path()
MODEL_METADATA = load_model_metadata(MODEL_NPZ_PATH)
CLASS_NAMES = MODEL_METADATA["class_names"]
DISPLAY_CLASS_NAMES = MODEL_METADATA["display_class_names"]
MODEL_RD_PATCH_HW = MODEL_METADATA["rd_patch_hw"]
MODEL_IQ_PATCH_HW = MODEL_METADATA["iq_patch_hw"]
MODEL_INPUT_STANDARDIZE = MODEL_METADATA["input_standardize"]


print(f"fs={fs}")
print(f"f_out={f_out}")
print(f"BW={BW}")
print(f"num_chirps={num_chirps}")
print(f"PRI_s={PRI_s}")
print(f"useable_point={useable_point}")
print(f"delay_point={delay_point}")

rpi_ip = "ip:phaser.local"
sdr_ip = "ip:192.168.2.1"
my_sdr = adi.ad9361(uri=sdr_ip)
my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)

my_phaser.configure(device_mode="rx")
my_phaser.element_spacing = 0.014
my_phaser.load_gain_cal()
my_phaser.load_phase_cal()
for i in range(0, 8):
    my_phaser.set_chan_phase(i, 0)

gain_list = [127] * 8
for i in range(0, len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

my_phaser._gpios.gpio_tx_sw = 0
my_phaser._gpios.gpio_vctrl_1 = 1
my_phaser._gpios.gpio_vctrl_2 = 1

my_sdr.sample_rate = int(fs)
my_sdr.rx_lo = int(center_freq)
my_sdr.rx_enabled_channels = [0, 1]
my_sdr.gain_control_mode_chan0 = 'manual'
my_sdr.gain_control_mode_chan1 = 'manual'
my_sdr.rx_hardwaregain_chan0 = int(rx_gain)
my_sdr.rx_hardwaregain_chan1 = int(rx_gain)

my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True
my_sdr.tx_hardwaregain_chan0 = -88
my_sdr.tx_hardwaregain_chan1 = int(0)

vco_freq = int(f_out + f0 + center_freq)
num_steps = int(ramp_time_us)
my_phaser.frequency = int(vco_freq / 4)
my_phaser.freq_dev_range = int(BW / 4)
my_phaser.freq_dev_step = int((BW / 4) / num_steps)
my_phaser.freq_dev_time = int(ramp_time_us)
my_phaser.delay_word = 4095
my_phaser.delay_clk = "PFD"
my_phaser.delay_start_en = 0
my_phaser.ramp_delay_en = 0
my_phaser.trig_delay_en = 0
my_phaser.ramp_mode = "single_sawtooth_burst"
my_phaser.sing_ful_tri = 0
my_phaser.tx_trig_en = 1
my_phaser.enable = 0

sdr_pins = adi.one_bit_adc_dac(sdr_ip)
sdr_pins.gpio_tdd_ext_sync = True
tdd = adi.tddn(sdr_ip)
sdr_pins.gpio_phaser_enable = True
tdd.enable = False
tdd.sync_external = True
tdd.startup_delay_ms = 0
tdd.frame_length_ms = PRI_ms
tdd.burst_count = num_chirps

tdd.channel[0].enable = True
tdd.channel[0].polarity = False
tdd.channel[0].on_raw = 0
tdd.channel[0].off_raw = 10
tdd.channel[1].enable = True
tdd.channel[1].polarity = False
tdd.channel[1].on_raw = 0
tdd.channel[1].off_raw = 10
tdd.channel[2].enable = True
tdd.channel[2].polarity = False
tdd.channel[2].on_raw = 0
tdd.channel[2].off_raw = 10
tdd.enable = True

NN = int(2**18)
fc = int(f0)
ts = 1 / float(fs)
t = np.arange(0, NN * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 0.9 * (i + 1j * q)


def setup_tx_with_retry(sdr, tx_iq, max_retry=3, wait_s=0.3):
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

        except OSError as e:
            last_err = e
            print(f"[警告] TX 初始化失敗 (attempt {attempt}/{max_retry}): {e}")
            time.sleep(wait_s)

        try:
            sdr.tx([tx_iq, tx_iq])
            print(f"TX buffer initialized (attempt {attempt}/{max_retry})")
            return
        except OSError as e:
            last_err = e
            print(f"[警告] TX 初始化失敗 (attempt {attempt}/{max_retry}): {e}")
            time.sleep(wait_s)

    raise RuntimeError(
        "TX buffer init failed after retries. Device may still be used by another process."
    ) from last_err


setup_tx_with_retry(my_sdr, iq)


def cleanup_tx_buffer():
    try:
        my_sdr.tx_destroy_buffer()
    except Exception:
        pass


atexit.register(cleanup_tx_buffer)

S = [512,1024,2048, 4096, 8192, 16384, 32768, 65536,131072]
eff = useable_point / S[-1]
for i in range(len(S) - 1, -1, -1):
    if S[i] <= useable_point:
        eff = S[i] / useable_point
        break

good_ramp_samples = int(useable_point * eff)
samples_to_drop = useable_point - good_ramp_samples
samples_drop_begin = samples_to_drop // 2
begin_offset_time = samples_drop_begin / fs
start_offset_time = tdd.channel[0].on_ms/1e3 + begin_offset_time
start_offset_samples = int(start_offset_time * fs)
print("good_ramp_samples = ", good_ramp_samples)

total_number_one_chip = int(tdd.frame_length_ms/1000*fs)
fft_size = S[0]
for k in S:
    if k >= total_number_one_chip:
        fft_size = k
        break
power = int(np.log2(fft_size))
print("fft_size =", fft_size)

total_time_ms = PRI_ms * num_chirps
print("Total Time for all Chirps: ", total_time_ms, "ms")

buffer_time = 0
power = 12

while total_time_ms > buffer_time:
    power += 1
    buffer_size = int(2**power)
    buffer_time = buffer_size / fs * 1000
    if power == 22:
        break

print("buffer_size:", buffer_size)
print("buffer_time:", buffer_time, "ms")

my_sdr.rx_buffer_size = buffer_size

num_bursts = tdd.burst_count
BW_eff = BW * eff

R_res = c / (2 * BW_eff)
v_res = lam / (2 * num_bursts * PRI_s)
R_max = c * (fs/2) / (2 * slope)
V_max = lam/(4*PRI_s)

print(f"num_bursts={num_bursts}")
print(f"R_res={R_res}")
print(f"v_res={v_res}")
print(f"R_max={R_max}")
print(f"V_max={V_max}")

def get_radar_data():
    my_phaser._gpios.gpio_burst = 0
    my_phaser._gpios.gpio_burst = 1
    my_phaser._gpios.gpio_burst = 0

    data = my_sdr.rx()
    sum_data = data[0] + data[1]
    usable_samples = num_bursts * N_frame
    sum_data = sum_data[:usable_samples]

    reshaped = sum_data.reshape(num_bursts, N_frame)

    rx_bursts = reshaped[:, start_offset_samples:
                            start_offset_samples + good_ramp_samples]

    return rx_bursts

def freq_process(data, use_mti=True, return_complex=False):
    if use_mti:
        chirp_current = data[:-1]
        chirp_next = data[1:]

        correlation = np.sum(chirp_current * np.conj(chirp_next), axis=1)
        angle_diff = np.angle(correlation)

        phase_correction = np.exp(-1j * angle_diff)[:, np.newaxis]

        data = chirp_next - chirp_current * phase_correction

    fft2_complex = np.fft.fftshift(np.fft.fft2(data))
    fft2 = np.abs(fft2_complex)

    range_doppler_data = 20 * np.log10(fft2 + 1e-10).T

    if return_complex:
        return range_doppler_data, fft2_complex.T
    return range_doppler_data

def simple_2d_cfar(rd_map_db, guard_r=1, guard_c=1, train_r=6, train_c=4, offset_db=6.0):
    power_map = 10 ** (rd_map_db / 10.0)

    outer_r = guard_r + train_r
    outer_c = guard_c + train_c

    outer_size = (2 * outer_r + 1, 2 * outer_c + 1)
    guard_size = (2 * guard_r + 1, 2 * guard_c + 1)

    outer_cells = outer_size[0] * outer_size[1]
    guard_cells = guard_size[0] * guard_size[1]
    train_cells = outer_cells - guard_cells

    if train_cells <= 0:
        return np.zeros_like(rd_map_db, dtype=bool)

    outer_sum = ndimage.uniform_filter(power_map, size=outer_size, mode='nearest') * outer_cells
    guard_sum = ndimage.uniform_filter(power_map, size=guard_size, mode='nearest') * guard_cells

    noise_est = (outer_sum - guard_sum) / train_cells
    threshold = noise_est * (10 ** (offset_db / 10.0))
    return power_map > threshold


def conv2d_same(x_chw, w_oihw, b_o):
    out_channels = w_oihw.shape[0]
    h, w = x_chw.shape[1], x_chw.shape[2]
    out = np.zeros((out_channels, h, w), dtype=np.float32)

    for oc in range(out_channels):
        acc = np.zeros((h, w), dtype=np.float32)
        for ic in range(w_oihw.shape[1]):
            acc += ndimage.correlate(x_chw[ic], w_oihw[oc, ic], mode='constant', cval=0.0)
        acc += b_o[oc]
        out[oc] = acc
    return out


def bn2d(x_chw, gamma, beta, mean, var, eps=1e-5):
    gamma = gamma[:, None, None]
    beta = beta[:, None, None]
    mean = mean[:, None, None]
    var = var[:, None, None]
    return ((x_chw - mean) / np.sqrt(var + eps)) * gamma + beta


def relu(x):
    return np.maximum(x, 0.0)


def maxpool2x2(x_chw):
    c, h, w = x_chw.shape
    h2 = h // 2
    w2 = w // 2
    x = x_chw[:, :h2 * 2, :w2 * 2]
    x = x.reshape(c, h2, 2, w2, 2)
    return x.max(axis=(2, 4))


def adaptive_avg_pool2d(x_chw, out_h, out_w):
    c, in_h, in_w = x_chw.shape
    out = np.zeros((c, out_h, out_w), dtype=np.float32)
    for oh in range(out_h):
        hs = int(np.floor(oh * in_h / out_h))
        he = int(np.ceil((oh + 1) * in_h / out_h))
        for ow in range(out_w):
            ws = int(np.floor(ow * in_w / out_w))
            we = int(np.ceil((ow + 1) * in_w / out_w))
            patch = x_chw[:, hs:he, ws:we]
            out[:, oh, ow] = np.mean(patch, axis=(1, 2))
    return out


def linear(x_1d, w_oi, b_o):
    return w_oi @ x_1d + b_o


def softmax(x_1d):
    z = x_1d - np.max(x_1d)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def branch_forward(x_chw, params, prefix):
    x = conv2d_same(x_chw, params[f"{prefix}.net.0.weight"], params[f"{prefix}.net.0.bias"])
    x = bn2d(x, params[f"{prefix}.net.1.weight"], params[f"{prefix}.net.1.bias"],
             params[f"{prefix}.net.1.running_mean"], params[f"{prefix}.net.1.running_var"])
    x = relu(x)
    x = maxpool2x2(x)

    x = conv2d_same(x, params[f"{prefix}.net.4.weight"], params[f"{prefix}.net.4.bias"])
    x = bn2d(x, params[f"{prefix}.net.5.weight"], params[f"{prefix}.net.5.bias"],
             params[f"{prefix}.net.5.running_mean"], params[f"{prefix}.net.5.running_var"])
    x = relu(x)
    x = maxpool2x2(x)

    x = conv2d_same(x, params[f"{prefix}.net.8.weight"], params[f"{prefix}.net.8.bias"])
    x = bn2d(x, params[f"{prefix}.net.9.weight"], params[f"{prefix}.net.9.bias"],
             params[f"{prefix}.net.9.running_mean"], params[f"{prefix}.net.9.running_var"])
    x = relu(x)
    if prefix == "rd_branch":
        x = adaptive_avg_pool2d(x, 2, 2)
    else:
        x = adaptive_avg_pool2d(x, 4, 2)
    return x


def radar_fusion_forward(rd_chw, iq_chw, params):
    rd_feat = branch_forward(rd_chw, params, "rd_branch")
    iq_feat = branch_forward(iq_chw, params, "iq_branch")
    fused = np.concatenate([rd_feat.reshape(-1), iq_feat.reshape(-1)]).astype(np.float32)

    expected_in = int(params["classifier.0.weight"].shape[1])
    if fused.size != expected_in:
        raise ValueError(
            f"feature length mismatch: got {fused.size}, expected {expected_in}. "
            f"Please check RD/IQ patch sizes."
        )

    x = relu(linear(fused, params["classifier.0.weight"], params["classifier.0.bias"]))
    x = relu(linear(x, params["classifier.3.weight"], params["classifier.3.bias"]))
    logits = linear(x, params["classifier.6.weight"], params["classifier.6.bias"])
    return logits


def crop_patch_like_selector(arr2d, center_x, center_y, out_h=16, out_w=16):
    half_h = out_h // 2
    half_w = out_w // 2

    y_start = max(0, center_y - half_h)
    x_start = max(0, center_x - half_w)
    y_end = min(arr2d.shape[0], y_start + out_h)
    x_end = min(arr2d.shape[1], x_start + out_w)

    patch = arr2d[y_start:y_end, x_start:x_end]
    out = np.zeros((out_h, out_w), dtype=arr2d.dtype)
    out[:patch.shape[0], :patch.shape[1]] = patch
    return out


def standardize_patch(x2d, clip_std=5.0):
    x = x2d.astype(np.float32, copy=False)
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-6:
        s = 1.0
    z = (x - m) / s
    if clip_std is not None and clip_std > 0:
        z = np.clip(z, -clip_std, clip_std)
    return z.astype(np.float32, copy=False)


def minmax_patch(x2d):
    x = x2d.astype(np.float32, copy=False)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max > x_min:
        return ((x - x_min) / (x_max - x_min)).astype(np.float32, copy=False)
    return np.zeros_like(x, dtype=np.float32)


def build_model_inputs(rd_map_db, rd_complex, center_x, center_y, patch_size=16):
    rd_h, rd_w = MODEL_RD_PATCH_HW
    iq_h, iq_w = MODEL_IQ_PATCH_HW

    rd_patch = crop_patch_like_selector(rd_map_db, center_x, center_y, rd_h, rd_w).astype(np.float32)
    if rd_complex is None:
        iq_real = np.zeros((iq_h, iq_w), dtype=np.float32)
        iq_imag = np.zeros((iq_h, iq_w), dtype=np.float32)
    else:
        iq_real = crop_patch_like_selector(np.real(rd_complex), center_x, center_y, iq_h, iq_w).astype(np.float32)
        iq_imag = crop_patch_like_selector(np.imag(rd_complex), center_x, center_y, iq_h, iq_w).astype(np.float32)

    if MODEL_INPUT_STANDARDIZE:
        rd_patch = standardize_patch(rd_patch, MODEL_CLIP_STD)
        iq_real = standardize_patch(iq_real, MODEL_CLIP_STD)
        iq_imag = standardize_patch(iq_imag, MODEL_CLIP_STD)
    else:
        rd_patch = minmax_patch(rd_patch)
        iq_real = minmax_patch(iq_real)
        iq_imag = minmax_patch(iq_imag)

    rd_chw = rd_patch[None, :, :]
    iq_chw = np.stack([iq_real, iq_imag], axis=0)
    return rd_chw, iq_chw


def load_radar_model_npz(npz_path):
    if not os.path.exists(npz_path):
        return None, f"model file not found: {npz_path}"

    try:
        data = np.load(npz_path, allow_pickle=False)
        key_map = json.loads(str(data["key_map_json"]))
        params = {}
        for saved_key, original_key in key_map:
            params[original_key] = data[saved_key].astype(np.float32, copy=False)
        return params, None
    except Exception as e:
        return None, str(e)


rx_bursts = get_radar_data()
radar_data = freq_process(rx_bursts)

model_params, model_error = load_radar_model_npz(MODEL_NPZ_PATH)
if model_params is None:
    print(f"[模型] 未啟用分類: {model_error}")
else:
    expected_in = int(model_params["classifier.0.weight"].shape[1])
    class_count = int(model_params["classifier.6.bias"].shape[0])
    if class_count != len(CLASS_NAMES):
        print(f"[模型] 類別數不一致: model={class_count}, config={len(CLASS_NAMES)}")
    class_map_str = ", ".join([f"{i}:{DISPLAY_CLASS_NAMES[i]}" for i in range(min(class_count, len(DISPLAY_CLASS_NAMES)))])
    print(
        f"[模型] 對齊設定: rd_patch={MODEL_RD_PATCH_HW}, iq_patch={MODEL_IQ_PATCH_HW}, "
        f"classifier_in={expected_in}"
    )
    print(f"[模型] 類別順序: {class_map_str}")
    print(f"[模型] 載入成功: {MODEL_NPZ_PATH}")

fig, ax = plt.subplots(figsize=(14, 7))
d_min_ext = dist.min()
d_max_ext = dist.max()

extent = [-V_max, V_max, dist.min(), dist.max()]
cmaps = ['inferno', 'plasma']
cmn = cmaps[0]

try:
    range_doppler_plot = ax.imshow(radar_data, aspect='auto',
        extent=extent, origin='lower', cmap=matplotlib.colormaps.get_cmap(cmn))
except:
    print("Using an older version of MatPlotLIB")
    from matplotlib.cm import get_cmap
    range_doppler_plot = ax.imshow(radar_data, aspect='auto',
        extent=extent, origin='lower', cmap=get_cmap(cmn))

ax.set_title('Range Doppler Spectrum', fontsize=24)
ax.set_xlabel('Velocity [m/s]', fontsize=22)
ax.set_ylabel('Range [m]', fontsize=22)
ax.set_xlim([-10, 10])
ax.set_ylim([0, 30])

ax.set_yticks(np.arange(0, 30, 30/20))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

print("sample_rate = ", fs/1e6, "MHz, ramp_time = ", ramp_time_us, "us, num_chirps = ", num_chirps)
print("CTRL + c to stop the loop")

data_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def data_worker():
    print("背景資料擷取與處理執行緒已啟動...")
    count = 0
    while not stop_event.is_set():
        try:
            rx_bursts = get_radar_data()
            radar_map, radar_complex = freq_process(rx_bursts, return_complex=True)
            
            if data_queue.full():
                data_queue.get()
            data_queue.put((radar_map, radar_complex))
            
        except Exception as e:
            print(f"\n[錯誤] 背景執行緒發生異常: {e}")
            break

worker_thread = threading.Thread(target=data_worker, daemon=True)
worker_thread.start()

print("\n主執行緒開始繪圖... (在終端機按 CTRL+C 停止)")

MAX_TRACKS = 30
track_artists = []
for _ in range(MAX_TRACKS):
    rect = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='red',
                             facecolor='none', linestyle='-', visible=False)
    ax.add_patch(rect)
    cross, = ax.plot([], [], 'rx', markersize=10, markeredgewidth=2, visible=False)
    txt = ax.text(0, 0, '', color='white', fontsize=10, fontweight='bold',
                  bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2),
                  visible=False)
    track_artists.append((rect, cross, txt))

plt.ion()  
plt.show() 

try:
    while True:
        try:
            latest_payload = data_queue.get(timeout=0.05)
            if isinstance(latest_payload, tuple):
                latest_radar_data, latest_radar_complex = latest_payload
            else:
                latest_radar_data = latest_payload
                latest_radar_complex = None

            num_r, num_c = latest_radar_data.shape

            range_axis_m = np.linspace(d_min_ext, d_max_ext, num_r)
            valid_range_rows = (range_axis_m >= range_min) & (range_axis_m <= range_max)

            if np.any(valid_range_rows):
                disp_max = np.max(latest_radar_data[valid_range_rows, :])
            else:
                disp_max = np.max(latest_radar_data)

            dynamic_range = 40
            vmin_dynamic = disp_max - dynamic_range
            vmax_dynamic = disp_max

            range_doppler_plot.set_data(latest_radar_data)
            range_doppler_plot.set_clim(vmin=vmin_dynamic, vmax=vmax_dynamic)

            binary_map = simple_2d_cfar(latest_radar_data)

            binary_map[~valid_range_rows, :] = False

            labeled_array, _ = ndimage.label(binary_map)
            objects = ndimage.find_objects(labeled_array)

            detections = []
            best_detection = None
            best_score = -np.inf

            for obj in objects:
                if obj is None:
                    continue
                r_slice, c_slice = obj

                r_min_idx, r_max_idx = r_slice.start, r_slice.stop - 1
                c_min_idx, c_max_idx = c_slice.start, c_slice.stop - 1

                v_min = -V_max + (c_min_idx / (num_c - 1)) * (2 * V_max)
                v_max = -V_max + (c_max_idx / (num_c - 1)) * (2 * V_max)
                r_min = d_min_ext + (r_min_idx / (num_r - 1)) * (d_max_ext - d_min_ext)
                r_max = d_min_ext + (r_max_idx / (num_r - 1)) * (d_max_ext - d_min_ext)

                w = v_max - v_min
                h = r_max - r_min

                dv = (2 * V_max) / (num_c - 1)
                dr = (d_max_ext - d_min_ext) / (num_r - 1)

                if w == 0: w = dv; v_min -= w/2
                if h == 0: h = dr; r_min -= h/2

                x = v_min - 1.5 * dv
                y = r_min - 1.5 * dr
                bw = w + 3 * dv
                bh = h + 3 * dr

                v_center = (v_min + v_max) / 2
                r_center = (r_min + r_max) / 2
                c_center_idx = int(round((c_min_idx + c_max_idx) / 2))
                r_center_idx = int(round((r_min_idx + r_max_idx) / 2))

                peak_db = np.max(latest_radar_data[r_slice, c_slice])
                if peak_db > best_score:
                    best_score = peak_db
                    best_detection = (x, y, bw, bh, v_center, r_center, c_center_idx, r_center_idx)

            if best_detection is not None:
                detections.append(best_detection)

            pred_label = ""
            pred_conf = 0.0
            if model_params is not None and best_detection is not None:
                try:
                    _, _, _, _, _, _, c_idx, r_idx = best_detection
                    rd_chw, iq_chw = build_model_inputs(
                        latest_radar_data,
                        latest_radar_complex,
                        c_idx,
                        r_idx,
                        patch_size=MODEL_PATCH_SIZE,
                    )
                    logits = radar_fusion_forward(rd_chw, iq_chw, model_params)
                    probs = softmax(logits)
                    pred_idx = int(np.argmax(probs))
                    pred_conf = float(probs[pred_idx] * 100.0)
                    if pred_idx < len(DISPLAY_CLASS_NAMES):
                        pred_label = DISPLAY_CLASS_NAMES[pred_idx]
                    elif pred_idx < len(CLASS_NAMES):
                        pred_label = CLASS_NAMES[pred_idx]
                    else:
                        pred_label = str(pred_idx)
                except Exception as e:
                    pred_label = f"ERR:{e}"

            for idx, (rect, cross, txt) in enumerate(track_artists):
                if idx < len(detections):
                    x, y, bw, bh, v_center, r_center, _, _ = detections[idx]
                    rect.set_bounds(x, y, bw, bh)
                    rect.set_visible(True)

                    cross.set_data([v_center], [r_center])
                    cross.set_visible(True)

                    txt.set_position((x + bw, r_center))
                    if pred_label:
                        txt.set_text(f' R: {r_center:.1f}m\n V: {v_center:.1f}m/s\n C: {pred_label} ({pred_conf:.1f}%)')
                    else:
                        txt.set_text(f' R: {r_center:.1f}m\n V: {v_center:.1f}m/s')
                    txt.set_visible(True)
                else:
                    rect.set_visible(False)
                    cross.set_visible(False)
                    txt.set_visible(False)

        except queue.Empty:
            pass

        plt.pause(0.01)

except KeyboardInterrupt:
    print("\n已接收到停止指令，正在安全關閉系統...")

finally:
    plt.ioff()
    plt.close('all')
    stop_event.set()
    worker_thread.join(timeout=1.0)
    my_sdr.tx_destroy_buffer()
    print("\nPluto Buffer Cleared & System Shutdown Complete!")