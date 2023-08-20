def optimize_miguel(buffer, criterion, batch_size, online_model, target_model, gamma, optimizer, epoch_loss):
    transitions = buffer.sample(batch_size)
    (states, actions, next_states, rewards, is_terminals) = online_model.load(transitions)
    # states, actions, rewards, next_states, is_terminals = experiences
    # batch_size = len(is_terminals)

    max_a_q_sp = target_model(next_states).detach().max(1)[0]  # (batch_size)
    target_q_sa = rewards + (gamma * max_a_q_sp * (1 - is_terminals))  # (batch_size)
    target_q_sa = target_q_sa.unsqueeze(1)  # (batch_size x 1)
    q_sa = online_model(states).gather(1, actions.unsqueeze(1))  # (batch_size x 1)

    td_error = q_sa - target_q_sa
    # value_loss = td_error.pow(2).mul(0.5).mean()
    value_loss = criterion(q_sa, target_q_sa)
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()
    epoch_loss.append(value_loss.data.cpu().numpy().copy().item())


def optimize_jim(buffer, criterion, batch_size, online_model, target_model, gamma, optimizer, epoch_loss):
    transitions = buffer.sample(batch_size)
    (states, actions, new_states, rewards, is_terminals) = online_model.load(transitions)
    continue_mask = 1 - is_terminals  # (batch_size)
    q_next = target_model(new_states).detach()  # gradient does NOT involve the target
    q_next_max = q_next.max(1)[0]  # (batch_size)
    q_target = rewards + q_next_max * continue_mask * gamma  # (batch_size)
    q_target = q_target.unsqueeze(1)  # (batch_size x 1)
    q_values = online_model(states).gather(1, actions.unsqueeze(1))  # (batch_size x 1)
    loss = criterion(q_values, q_target)
    epoch_loss.append(loss.data.cpu().numpy().copy().item())
    # may want to return this loss if it has no access to self.epoch_loss
    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()