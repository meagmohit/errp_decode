import numpy as np
import matplotlib.pyplot as plt

plt.xlabel('Episode')
plt.ylabel('Success Rate')
max_episode = 1000

comp_mean, comp_std = [], []


filenames = ['noErrP_0.npz', 'noErrP_1.npz', 'noErrP_2.npz', 'noErrP_3.npz', 'noErrP_4.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p0,  = plt.plot(res_m, color='gray')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='lightgray', alpha=0.5)
print("No ErrP alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))


filenames = ['p090_0.npz', 'p090_1.npz', 'p090_2.npz', 'p090_3.npz', 'p090_4.npz']#,'qmaze_all_sub01_5.npz', 'qmaze_all_sub01_6.npz', 'qmaze_all_sub01_7.npz', 'qmaze_all_sub01_8.npz', 'qmaze_all_sub01_9.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']#[:, 0]
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p1,  = plt.plot(res_m, color='blue')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='lightblue', alpha=0.5)
print("sub01 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p085_0.npz', 'p085_1.npz','p085_2.npz', 'p085_3.npz', 'p085_4.npz']#,
             #'qmaze_all_sub03_5.npz', 'qmaze_all_sub03_6.npz', 'qmaze_all_sub03_7.npz', 'qmaze_all_sub03_8.npz', 'qmaze_all_sub03_9.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']#[:, 0]
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p2,  = plt.plot(res_m, color='red')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='lightcoral', alpha=0.5)
print("sub02 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p080_0.npz', 'p080_1.npz', 'p080_2.npz', 'p080_3.npz', 'p080_4.npz']#
             #'qmaze_all_sub04_5.npz', 'qmaze_all_sub04_6.npz', 'qmaze_all_sub04_7.npz', 'qmaze_all_sub04_8.npz', 'qmaze_all_sub04_9.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']#[:, 0]
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p3,  = plt.plot(res_m, color='green')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='lightgreen', alpha=0.5)
print("sub06 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p075_0.npz', 'p075_1.npz', 'p075_2.npz', 'p075_3.npz', 'p075_4.npz']
             #'qmaze_all_sub07_5.npz', 'qmaze_all_sub07_6.npz', 'qmaze_all_sub07_7.npz', 'qmaze_all_sub07_8.npz', 'qmaze_all_sub07_9.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']#[:, 0]
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p4,  = plt.plot(res_m, color='yellow')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='gold', alpha=0.5)
print("sub07 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p070_0.npz', 'p070_1.npz', 'p070_2.npz', 'p070_3.npz', 'p070_4.npz']
 #filenames = ['qmaze_all_sub09_0.npz', 'qmaze_all_sub09_1.npz', 'qmaze_all_sub09_2.npz', 'qmaze_all_sub09_3.npz', 'qmaze_all_sub09_4.npz',
 #             'qmaze_all_sub09_5.npz', 'qmaze_all_sub09_6.npz', 'qmaze_all_sub09_7.npz', 'qmaze_all_sub09_8.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']#[:, 0]
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p5,  = plt.plot(res_m, color='cyan')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='lightcyan', alpha=0.5)
print("sub16 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p065_0.npz', 'p065_1.npz', 'p065_2.npz', 'p065_3.npz', 'p065_4.npz']
 #filenames = ['qmaze_all_sub09_0.npz', 'qmaze_all_sub09_1.npz', 'qmaze_all_sub09_2.npz', 'qmaze_all_sub09_3.npz', 'qmaze_all_sub09_4.npz',
 #             'qmaze_all_sub09_5.npz', 'qmaze_all_sub09_6.npz', 'qmaze_all_sub09_7.npz', 'qmaze_all_sub09_8.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p6,  = plt.plot(res_m, color='orchid')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='plum', alpha=0.5)
print("sub16 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p060_0.npz', 'p060_1.npz', 'p060_2.npz', 'p060_3.npz', 'p060_4.npz']
 #filenames = ['qmaze_all_sub09_0.npz', 'qmaze_all_sub09_1.npz', 'qmaze_all_sub09_2.npz', 'qmaze_all_sub09_3.npz', 'qmaze_all_sub09_4.npz',
 #             'qmaze_all_sub09_5.npz', 'qmaze_all_sub09_6.npz', 'qmaze_all_sub09_7.npz', 'qmaze_all_sub09_8.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p7,  = plt.plot(res_m, color='crimson')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='lavenderblush', alpha=0.5)
print("sub16 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p055_0.npz', 'p055_1.npz', 'p055_2.npz', 'p055_3.npz', 'p055_4.npz']
 #filenames = ['qmaze_all_sub09_0.npz', 'qmaze_all_sub09_1.npz', 'qmaze_all_sub09_2.npz', 'qmaze_all_sub09_3.npz', 'qmaze_all_sub09_4.npz',
 #             'qmaze_all_sub09_5.npz', 'qmaze_all_sub09_6.npz', 'qmaze_all_sub09_7.npz', 'qmaze_all_sub09_8.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p8,  = plt.plot(res_m, color='saddlebrown')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='sandybrown', alpha=0.5)
print("sub16 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

filenames = ['p050_0.npz', 'p050_1.npz', 'p050_2.npz', 'p050_3.npz', 'p050_4.npz']
 #filenames = ['qmaze_all_sub09_0.npz', 'qmaze_all_sub09_1.npz', 'qmaze_all_sub09_2.npz', 'qmaze_all_sub09_3.npz', 'qmaze_all_sub09_4.npz',
 #             'qmaze_all_sub09_5.npz', 'qmaze_all_sub09_6.npz', 'qmaze_all_sub09_7.npz', 'qmaze_all_sub09_8.npz']
comp_ep = []
res = np.ones((len(filenames), max_episode))
for k, fn in enumerate(filenames):
    data = np.load(fn)['res']
    res[k, :np.size(data)] = data.reshape(-1)
    comp_ep.append(np.size(data))
comp_ep = np.array(comp_ep)
res_v = np.var(res, axis=0)
res_m = np.mean(res, axis=0)
res_up = np.clip(res_m + 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
res_dw = np.clip(res_m - 1.96/np.sqrt(len(filenames))*np.sqrt(res_v), 0, 1.)
p9,  = plt.plot(res_m, color='orangered')
plt.fill_between(np.arange(max_episode), res_up, res_dw, color='lightsalmon', alpha=0.5)
print("sub16 alg. complete episode: {:.1f} ({:.1f})".format(np.mean(comp_ep), np.std(comp_ep)))
print(comp_ep)
comp_mean.append(np.mean(comp_ep))
comp_std.append(np.std(comp_ep))

plt.legend([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9], ['No ErrP', 'p=0.9', 'p=0.85', 'p=0.8', 'p=0.75', 'p=0.7', 'p=0.65', 'p=0.6', 'p=0.55', 'p=0.5'], loc='lower right')
plt.show()

# plot error bar
fig, ax = plt.subplots()
colors = ['blue', 'red', 'green', 'yellow', 'cyan']

container = ax.bar(['NoErrP', 'p=0.9','p=0.85', 'p=0.8', 'p=0.75', 'p=0.7', 'p=0.65', 'p=0.6', 'p=0.55', 'p=0.5'], comp_mean, yerr=comp_std,
                   alpha=0.5, color=colors, error_kw=dict(lw=3, capsize=4, capthick=2))
ax.margins(0.05)

np.savez('full_access', mean=np.array(comp_mean), std=np.array(comp_std))
connector, caplines, (vertical_lines,) = container.errorbar.lines
vertical_lines.set_color(colors)
plt.ylabel('Complete Episode')
plt.xlabel('ErrP Detection Probability')
plt.show()
