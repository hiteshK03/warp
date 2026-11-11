import warp as wp

m = wp.Module("q5_kernels")
wp.set_module_options({"metal_float64": "df64"}, module=m)


@wp.kernel(module=m)
def good_0(a: wp.array[float], o: wp.array[float], c: float):
    i = wp.tid()
    o[i] = a[i] * c + 0.0


@wp.kernel(module=m)
def good_1(a: wp.array[float], o: wp.array[float], c: float):
    i = wp.tid()
    o[i] = a[i] * c + 1.0


@wp.kernel(module=m)
def good_2(a: wp.array[float], o: wp.array[float], c: float):
    i = wp.tid()
    o[i] = a[i] * c + 2.0


@wp.kernel(module=m)
def good_3(a: wp.array[float], o: wp.array[float], c: float):
    i = wp.tid()
    o[i] = a[i] * c + 3.0


@wp.kernel(module=m)
def good_4(a: wp.array[float], o: wp.array[float], c: float):
    i = wp.tid()
    o[i] = a[i] * c + 4.0


@wp.kernel(module=m)
def good_5(a: wp.array[float], o: wp.array[float], c: float):
    i = wp.tid()
    o[i] = a[i] * c + 5.0


@wp.kernel(module=m)
def bad(o: wp.array[float]):
    o[0] = wp.float32(wp.sin(wp.float64(0.5)))


goods = [good_0, good_1, good_2, good_3, good_4, good_5]
