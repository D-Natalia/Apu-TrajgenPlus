import matplotlib.pyplot as plt

intervale = ["80-155", "170-350", "250-450", "350-550"]

su1 = [0.9639, 0.9650, 0.9850, 0.6030]
su2 = [0.9070, 0.8070, 0.6670, 0.0960]
su3 = [0.9800, 0.9950, 0.9630, 0.2070]
su5 = [0.8840, 0.8239, 0.6980, 0.0930]

plt.figure(figsize=(8,5))

plt.plot(intervale, su1, marker='o',markersize=8, linewidth=2.5, label='SU1')
plt.plot(intervale, su2, marker='s', markersize=8, linewidth=2.5, label='SU2')
plt.plot(intervale, su3, marker='^', markersize=8, linewidth=2.5, label='SU3')
plt.plot(intervale, su5, marker='d', markersize=8, linewidth=2.5, label='SU5')

plt.xlabel("Intervalul de protecție (m)")
plt.ylabel("Acuratețe")
plt.title("Comparația acurateții metodelor")
plt.ylim(0,1.05)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("comparatie_accuracy.png", dpi=300)
plt.show()