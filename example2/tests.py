# coding=utf-8
from unittest import TestCase
from example.exmath import exsum
from example.exencode import amount


class ExTestCase(TestCase):
    def test_exsum(self):
        self.assertEqual(exsum(3,2), 5)


class EncTestCase(TestCase):
    def test_amount(self):
        self.assertEqual(amount('10'), '€ 10.00')
