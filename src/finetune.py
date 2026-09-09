import torch


def fine_tune(
    model,
    train_loader,
    device,
    epochs=2,
    learning_rate=0.0001,
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
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total

        print(
            f"Fine-tune epoch [{epoch + 1}/{epochs}] "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )