
cdef inline v(double[:] line, int offset):
    """V型拐点"""
    cdef double a, b, c
    a, b, c = line[offset - 2], line[offset - 1], line[offset]
    return b <= a and b < c


cdef inline vdown(double[:] line, int offset):
    """/\型拐点"""
    cdef double a, b, c
    a, b, c = line[offset - 2], line[offset - 1], line[offset]
    return b >= a and b > c


cdef inline kup(double[:] line, int offset):
    """斜率向上"""
    return line[offset] > line[offset - 1]


cdef inline kdown(double[:] line, int offset):
    """斜率向下"""
    return line[offset] < line[offset - 1]


cdef inline ne_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] != line2[offset]"""
    return line1[offset] != line2[offset]


cdef inline eq_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] == line2[offset]"""
    return line1[offset] == line2[offset]


cdef inline lt_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] < line2[offset]"""
    return line1[offset] < line2[offset]


cdef inline gt_l(double[:] line1, double[:] line2):
    """line1[offset] > line2[offset]"""
    return line1[-1] > line2[-1]


cdef inline lte_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] <= line2[offset]"""
    return line1[offset] <= line2[offset]


cdef inline gte_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] >= line2[offset]"""
    return line1[offset] >= line2[offset]


cdef inline ne(double[:] line, double num, int offset):
    """line[offset] != num"""
    return line[offset] != num


cdef inline eq(double[:] line, double num, int offset):
    """line[offset] == num"""
    return line[offset] == num


cdef inline lt(double[:] line, double num, int offset):
    """line[offset] < num"""
    return line[offset] < num


cdef inline gt(double[:] line, double num, int offset):
    """line[offset] > num"""
    return line[offset] > num

cdef inline lte(double[:] line, double num, int offset):
    """line[offset] <= num"""
    return line[offset] <= num


cdef inline gte(double[:] line, double num, int offset):
    """line[offset] >= num"""
    return line[offset] >= num


cdef inline crossup(double[:] line1, double[:] line2, int offset):
    """金叉: line1 上穿 line2"""
    cdef int pre_offset = offset - 1
    return (line1[offset] > line2[offset]) and (line1[pre_offset] <= line2[pre_offset])


cdef inline crossdown(double[:] line1, double[:] line2, int offset):
    """死叉: line1 下穿 line2"""
    cdef int pre_offset = offset - 1
    return (line1[offset] < line2[offset]) and (line1[pre_offset] >= line2[pre_offset])


cdef inline crossup_value(double[:] line, double value, int offset):
    """上穿value"""
    return (line[offset] > value) and (line[offset - 1] <= value)


cdef inline crossdown_value(double[:] line, double value, int offset):
    """下穿value"""
    return (line[offset] < value) and (line[offset - 1] >= value)


cdef inline openup(double[:] line1, double[:] line2, int offset):
    """开口增大"""
    cdef:
        int pre_offset
        double n1, n2, m1, m2
    pre_offset = offset - 1
    n1, n2 = line1[offset], line1[pre_offset]
    m1, m2 = line2[offset], line2[pre_offset]
    if n1 > m1 and n2 > m2:
        return (n1 - m1) > (n2 - m2)
    elif n1 < m1 and n2 < m2:
        return (m1 - n1) > (m2 - n2)
    else:
        return False


cdef inline opendown(double[:] line1, double[:] line2, int offset):
    """开口缩小"""
    cdef:
        int pre_offset
        double n1, n2, m1, m2
    pre_offset = offset - 1
    n1, n2 = line1[offset], line1[pre_offset]
    m1, m2 = line2[offset], line2[pre_offset]
    if n1 > m1 and n2 > m2:
        return (n1 - m1) < (n2 - m2)
    elif n1 < m1 and n2 < m2:
        return (m1 - n1) < (m2 - n2)
    else:
        return False


cdef inline red_bar(double[:] OPEN, double[:] CLOSE, int offset):
    """ 阳线，如果是十字星并且今收>昨收，也当成阳线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c > o or (c == o and c > pre_c)


cdef inline red_bar_real(double[:] OPEN, double[:] CLOSE, int offset):
    """ 上涨线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c > pre_c or (c == pre_c and c > o)


cdef inline green_bar(double[:] OPEN, double[:] CLOSE, int offset):
    """ 阴线，如果是十字星并且今收<=昨收，也当成阴线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c < o or (c == o and c <= pre_c)


cdef inline green_bar_real(double[:] OPEN, double[:] CLOSE, int offset):
    """ 下跌线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c < pre_c or (c == pre_c and c <= o)
