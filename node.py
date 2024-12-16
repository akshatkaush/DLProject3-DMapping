import torch


class Node:
    def __init__(
        self,
        dataloader,
        model_class,
        optimizer,
        loss_fn,
        lr=1e-3,
        device="cpu",
        rho=0.5,
        model_kwargs={},
    ):
        self.device = device
        self.model = model_class(**model_kwargs).to(device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn()
        self.dataloader = dataloader
        self.rho = rho
        self.iterer = iter(self.dataloader)
        self.history = {"pred_loss": [], "total_loss": [], "accuracy": []}
        self.dual = torch.zeros_like(
            torch.nn.utils.parameters_to_vector(self.model.parameters())
        )

    def primal_loss(self, dual, th, th_reg, pred_loss):
        reg = torch.sum(torch.square(th - th_reg.squeeze()))
        loss = pred_loss + torch.dot(th, dual) + self.rho * reg
        return loss

    def primal_step(self, th_reg, dual, iters=2):
        for _ in range(iters):
            print("Primal step:", _)
            self.optimizer.zero_grad()
            x, y = self.get_next_batch()
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            pred_loss = self.loss_fn(pred, y)
            th = torch.nn.utils.parameters_to_vector(self.model.parameters())
            loss = self.primal_loss(dual, th, th_reg, pred_loss)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.history["pred_loss"].append(pred_loss.item())
            self.history["total_loss"].append(loss.item())
            self.history["accuracy"].append(
                (pred.argmax(dim=1) == y).float().mean().item()
            )

    def get_next_batch(self):
        try:
            return next(self.iterer)
        except StopIteration:
            self.iterer = iter(self.dataloader)
            return next(self.iterer)
