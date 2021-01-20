from .smart_coord import smart_coord_test


def smart_coordinator(args, model1, model2, upan, device, test_loaders):
    result = []
    print(f"PAN type: {args.upan_type}")
    test_loss, acc = smart_coord_test(
        args, model1, model2, upan, device, test_loaders
    )
    result.append({"test_loss": test_loss, "acc": acc})
    return result
