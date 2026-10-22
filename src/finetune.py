import torch


def fine_tune(
    model,
    train_loader,
    device,
    epochs=2,
    learning_rate=0.0001,
    progress_callback=None,
    candidate_index=None,
    candidate_total=None,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        accuracy = (
            100.0 * correct / total
        )

        if progress_callback is not None:
            progress_callback(
                stage="fine_tuning",
                candidate=candidate_index,
                total_candidates=candidate_total,
                epoch=epoch + 1,
                total_epochs=epochs,
                accuracy=accuracy,
                loss=(
                    total_loss /
                    len(train_loader)
                ),
            )