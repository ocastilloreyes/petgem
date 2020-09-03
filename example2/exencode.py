# coding=utf-8
EUR = "€"


def amount(tariff, currency=EUR):
    return '{0} {1:.2f}'.format(currency, float(tariff))
