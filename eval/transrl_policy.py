"""Cached Transformer-policy inference."""
import torch

class FastPolicy:

    def __init__(self, model, batch_size, feature_dim, graphs=True):
        self.model = model
        self.b = batch_size
        device = next(model.parameters()).device
        self.obs = torch.zeros((self.b, model.n, feature_dim), device=device)
        self.loads = torch.zeros((self.b, model.m), device=device)
        self.actions = torch.zeros((self.b, 2), device=device, dtype=torch.long)
        self.mask = torch.triu(torch.ones(2, 2, device=device, dtype=torch.bool), diagonal=1)
        self.graphs = graphs and device.type == 'cuda'
        with torch.inference_mode():
            if self.graphs:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.route()
                        self.sequence()
                torch.cuda.current_stream().wait_stream(stream)
                self.g0 = torch.cuda.CUDAGraph()
                self.g1 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.g0):
                    self.route()
                with torch.cuda.graph(self.g1):
                    self.sequence()
            else:
                self.route()
                self.sequence()

    def decode_role(self, role):
        m = self.model
        shifted = torch.full((self.b, 2), 5, dtype=torch.long, device=self.obs.device)
        shifted[:, 1] = self.actions[:, 0]
        q = m.action_embed(shifted) + self.role_mem
        reasoning = m.reasoning(m.decoder(q, self.memory, tgt_mask=self.mask, tgt_is_causal=True))
        return m.actors[role](reasoning[:, role]).log_softmax(-1)

    def route(self):
        m = self.model
        tasks = m.input(self.obs)
        role = m.roles[None] + m.load(self.loads)[:, None]
        self.memory = m.encoder(torch.cat([tasks, role], 1))
        self.role_mem = self.memory[:, -2:]
        self.value = m.critic(self.role_mem.mean(1)).squeeze(-1)
        self.lp0 = self.decode_role(0)

    def sequence(self):
        self.lp1 = self.decode_role(1)

    @torch.inference_mode()
    def act(self, obs, loads, generators):
        self.obs.copy_(obs)
        self.loads.copy_(loads)
        self.actions.zero_()
        if self.graphs:
            self.g0.replay()
        else:
            self.route()
        chosen = torch.stack([torch.multinomial(self.lp0[j].exp(), 1, generator=g)[0] for j, g in enumerate(generators)])
        self.actions[:, 0] = chosen
        lp0 = self.lp0.gather(1, chosen[:, None]).squeeze(1)
        if self.graphs:
            self.g1.replay()
        else:
            self.sequence()
        chosen = torch.stack([torch.multinomial(self.lp1[j].exp(), 1, generator=g)[0] for j, g in enumerate(generators)])
        self.actions[:, 1] = chosen
        lp1 = self.lp1.gather(1, chosen[:, None]).squeeze(1)
        return (self.actions.clone(), torch.stack([lp0, lp1], 1), self.value.clone())
