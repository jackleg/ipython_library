#!/usr/bin/env python
# -*- coding: utf8 -*-

def str_join(data_list, sep="\t"):
    """data_list들을 하나의 문자열로 만든다.

    str.join()과 유사하게 동작한다.
    단, data_list가 str이 아닌 경우에 사용 가능하다.
    
    :param data_list: join해서 하나로 만들 데이터 리스트.
    :param sep: data_list의 element들을 join할 때 사용할 glue str.
    :return: sep로 data_list를 붙인 하나의 str.
    """

    if not data_list: return ""

    str_list = map(str, data_list)
    return sep.join(str_list)
