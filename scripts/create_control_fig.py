#!/usr/bin/python
import matplotlib.pyplot as plt

f=open('esat/pilot_data/esatv3_expert_stochastic/gau/00000_esatv3/applied_info.txt','r')
lines=f.readlines()
f.close()
gau_ctr=[float(l.strip().split(' ')[-1]) for l in lines[2:]][::2]

f=open('esat/pilot_data/esatv3_expert_stochastic/gau/00000_esatv3/control_info.txt','r')
lines=f.readlines()
f.close()
gau_lbls=[float(l.strip().split(' ')[-1]) for l in lines[2:]][::2]
gau_ctr=gau_ctr[1:]

f=open('esat/pilot_data/esatv3_expert_stochastic/uni/00000_esatv3/control_info.txt','r')
lines=f.readlines()
f.close()
uni_lbls=[float(l.strip().split(' ')[-1]) for l in lines[2:]][::2]

f=open('esat/pilot_data/esatv3_expert_stochastic/uni/00000_esatv3/applied_info.txt','r')
lines=f.readlines()
f.close()
uni_ctr=[float(l.strip().split(' ')[-1]) for l in lines[2:]][::2]
uni_ctr=uni_ctr[1:]

f=open('esat/pilot_data/esatv3_expert_stochastic/ou/00000_esatv3/applied_info.txt','r')
lines=f.readlines()
f.close()
ou_ctr=[float(l.strip().split(' ')[-1]) for l in lines[2:]][::2]
f=open('esat/pilot_data/esatv3_expert_stochastic/ou/00000_esatv3/control_info.txt','r')
lines=f.readlines()
f.close()
ou_lbls=[float(l.strip().split(' ')[-1]) for l in lines[2:]][::2]
ou_ctr=ou_ctr[1:]

f, (ax3, ax2, ax1) = plt.subplots(1, 3, figsize=(15,5), sharey=True)

ax1.plot(ou_lbls[20:40], label='control')
ax1.plot(ou_ctr[20:40], label='control+noise')
ax1.set_title('Ornstein-Uhlenbeck')
ax1.legend()

ax2.plot(uni_lbls[20:40], label='control')
ax2.plot(uni_ctr[20:40], label='control+noise')
ax2.set_title('Uniform')
ax2.legend()

ax3.plot(gau_lbls[20:40], label='control')
ax3.plot(gau_ctr[20:40], label='control+noise')
ax3.set_title('Gaussian')
ax3.legend()

f.savefig('noisy_control.png')