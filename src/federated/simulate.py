import argparse
from typing import List
import os
import torch
import flwr as fl

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.models.cnn import build_model as build_cnn
from src.models.improved_cnn import build_model as build_improved_cnn
from src.utils.training import train_one_epoch, evaluate


def get_client_fn(cfg_path: str):
	cfg = load_config(cfg_path)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def client_fn(cid: str):
		client_id = int(cid)
		trainloader, valloader, class_weights = create_dataloaders(
			root=cfg.train.data_root,
			image_size=cfg.train.image_size,
			class_names=cfg.train.class_names,
			batch_size=cfg.train.batch_size,
			num_workers=cfg.train.num_workers,
			val_split=cfg.train.val_split,
			seed=cfg.train.seed + client_id,
		)
		
		# Use pre-trained model if specified
		if hasattr(cfg.train, 'use_pretrained') and cfg.train.use_pretrained:
			model = build_improved_cnn(num_classes=cfg.train.num_classes).to(device)
		else:
			model = build_cnn(num_classes=cfg.train.num_classes).to(device)

		class SimClient(fl.client.NumPyClient):
			def __init__(self):
				self.model = model
				self.trainloader = trainloader
				self.valloader = valloader
				self.device = device
				self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

			def get_parameters(self, config):
				return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

			def set_parameters(self, parameters) -> None:
				state_dict = self.model.state_dict()
				for (k, _), v in zip(state_dict.items(), parameters):
					state_dict[k] = torch.tensor(v)
				self.model.load_state_dict(state_dict, strict=True)

			def fit(self, parameters, config):
				self.set_parameters(parameters)
				train_loss, train_acc = train_one_epoch(self.model, self.trainloader, self.optimizer, self.device, class_weights)
				return self.get_parameters({}), len(self.trainloader.dataset), {"train_loss": train_loss, "train_acc": train_acc}

			def evaluate(self, parameters, config):
				self.set_parameters(parameters)
				val_loss, val_acc = evaluate(self.model, self.valloader, self.device, class_weights)
				return float(val_loss), len(self.valloader.dataset), {"val_acc": float(val_acc)}

		return SimClient()

	return client_fn


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/optimized.yaml")
	args = parser.parse_args()

	cfg = load_config(args.config)

	strategy = fl.server.strategy.FedAvg(
		fraction_fit=cfg.federated.fraction_fit,
		fraction_evaluate=cfg.federated.fraction_eval,
	)

	client_fn = get_client_fn(args.config)
	fl.simulation.start_simulation(
		client_fn=client_fn,
		num_clients=cfg.federated.num_clients,
		config=fl.server.ServerConfig(num_rounds=cfg.federated.rounds),
		strategy=strategy,
	)


if __name__ == "__main__":
	main()
