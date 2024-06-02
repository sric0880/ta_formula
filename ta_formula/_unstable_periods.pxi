# m = int((np.log(0.001) -np.log(2/(n+1))) / np.log((n - 1) / (n + 1)))
# n->m: m天前的价格对当天ema(n)的影响降低到0.001以下
cdef unsigned int[251] ema_unstable_periods = [0, 1, 5, 8, 11, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34, 36, 38, 39, 41, 43, 45, 47, 49, 50, 52, 54, 55, 57, 59, 60, 62, 64, 65, 67, 68, 70, 71, 73, 74, 76, 77, 79, 80, 82, 83, 84, 86, 87, 89, 90, 91, 93, 94, 95, 97, 98, 99, 100, 102, 103, 104, 105, 107, 108, 109, 110, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 171, 172, 173, 174, 175, 176, 177, 178, 178, 179, 180, 181, 182, 183, 183, 184, 185, 186, 187, 188, 188, 189, 190, 191, 192, 192, 193, 194, 195, 196, 196, 197, 198, 199, 200, 200, 201, 202, 203, 203, 204, 205, 206, 206, 207, 208, 209, 209, 210, 211, 211, 212, 213, 214, 214, 215, 216, 216, 217, 218, 219, 219, 220, 221, 221, 222, 223, 223, 224, 225, 225, 226, 227, 227, 228, 229, 229, 230, 231, 231, 232, 232, 233, 234, 234, 235, 236, 236, 237, 238, 238, 239, 239, 240, 241, 241, 242, 242, 243, 244, 244, 245, 245, 246, 247, 247, 248, 248, 249, 249, 250, 251, 251, 252, 252, 253, 253, 254, 255, 255, 256, 256, 257, 257, 258, 258, 259]