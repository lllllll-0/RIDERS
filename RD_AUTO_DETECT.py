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
import sys
import time
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
print("requested freq dev time (us) = ", ramp_time_us)
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
iq = 0.9* (i + 1j * q)


def setup_tx_with_retry(sdr, tx_iq, max_retry=3, wait_s=0.3):
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

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

print(f"num_bursts={num_bursts}  # burst數量")
print(f"R_res={R_res}  # 距離解析度(m)")
print(f"v_res={v_res}  # 速度解析度(m/s)")
print(f"R_max={R_max}  # 最大作用距離(m)")
print(f"V_max={V_max}  # 最大速度(m/s)")

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

def freq_process(data, use_mti=True):
    if use_mti:
        chirp_current = data[:-1]
        chirp_next = data[1:]

        correlation = np.sum(chirp_current * np.conj(chirp_next), axis=1)
        angle_diff = np.angle(correlation)

        phase_correction = np.exp(-1j * angle_diff)[:, np.newaxis]

        data = chirp_next - chirp_current * phase_correction

    fft2 = np.fft.fftshift(np.abs(np.fft.fft2(data)))

    range_doppler_data = 20 * np.log10(fft2 + 1e-10).T

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

rx_bursts = get_radar_data()
radar_data = freq_process(rx_bursts)

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
            radar_map = freq_process(rx_bursts)

            if data_queue.full():
                data_queue.get()
            data_queue.put(radar_map)

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
                  bbox=dict(facecolor='dimgray', alpha=0.7, edgecolor='none', pad=2),
                  visible=False)
    track_artists.append((rect, cross, txt))

plt.ion()  
plt.show() 

try:
    while True:
        try:
            latest_radar_data = data_queue.get(timeout=0.05)

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

                peak_db = np.max(latest_radar_data[r_slice, c_slice])
                if peak_db > best_score:
                    best_score = peak_db
                    best_detection = (x, y, bw, bh, v_center, r_center)

            if best_detection is not None:
                detections.append(best_detection)

            for idx, (rect, cross, txt) in enumerate(track_artists):
                if idx < len(detections):
                    x, y, bw, bh, v_center, r_center = detections[idx]
                    rect.set_bounds(x, y, bw, bh)
                    rect.set_visible(True)

                    cross.set_data([v_center], [r_center])
                    cross.set_visible(True)

                    txt.set_position((x + bw, r_center))
                    warning_text = ''
                    if abs(v_center) > 2.5:
                        warning_text = '\n WARNING: speed out of range'
                    txt.set_text(f' R: {r_center:.1f}m\n V: {v_center:.1f}m/s{warning_text}')
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