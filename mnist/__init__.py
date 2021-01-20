def smart_coordinator(args, model1, model2, upan, device, test_loader):
    result = []
    print(f"PAN type: {args.upan_type}")
    test_loss, acc = smart_coord_test(
        args, model1, model2, upan, device, test_loader
    )
    result.append({"test_loss": test_loss, "acc": acc})
    return result
