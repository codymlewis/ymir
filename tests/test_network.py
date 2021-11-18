from absl.testing import absltest

import ymir

class TestNetwork(absltest.TestCase):
    def test_controller(self):
        controller = ymir.mp.network.Controller(0.1)
        self.assertListEqual(controller.clients, [])
        self.assertListEqual(controller.switches, [])
        self.assertEqual(controller.C, 0.1)
        self.assertEqual(controller.K, 0)

    def test_network(self):
        network = ymir.mp.network.Network(0.1)
        self.assertListEqual(network.clients, [])
        self.assertDictEqual(network.controllers, {})
        self.assertEqual(network.server_name, "")
        self.assertEqual(network.C, 0.1)


if __name__ == '__main__':
    absltest.main()