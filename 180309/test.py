class test(object):
	def __init__(self):
		self.x = None
		self.y = None

	def input(self, x, y, z):
		self.x = x
		self.y = y
		localz = z

	def localz(self):
		print("z")

	def printz(self):
		print(localz)

Test = test()
Test.input(1,2,3)
Test.printz()
