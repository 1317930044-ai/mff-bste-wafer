"""Exercise Streamlit widgets and verify the selected sample against core inference."""

from streamlit.testing.v1 import AppTest

from wafer.common import ROOT, labels
from wafer.inference import available_models, available_samples, load_model, predict


def main():
    app = AppTest.from_file(str(ROOT / "web_app.py"), default_timeout=45).run()
    assert not app.exception, app.exception
    task = "defect8" if available_models("defect8") else "all9"
    models = available_models(task)
    if not models:
        raise SystemExit("请先运行 python main.py train-all --synthetic --mode quick --run-id demo-synthetic")
    next(radio for radio in app.radio if radio.label == "分类任务").set_value(task).run()
    sample = next(row for row in available_samples() if row["label"] in labels(task))
    path, _ = models[0]
    _, expected = predict(load_model(path), path=sample["path"])
    next(button for button in app.button if button.label == "识别晶圆").click().run()
    assert not app.exception, app.exception
    assert next(metric for metric in app.metric if metric.label == "预测类别").value == expected["prediction"]
    next(button for button in app.button if button.label == "批量识别").click().run()
    assert not app.exception, app.exception
    assert any(element.label == "下载批量 CSV" for element in app.get("download_button"))
    assert any(metric.label == "阶段" for metric in app.metric)
    next(radio for radio in app.radio if radio.label == "输入方式").set_value("上传文件").run()
    png = sample["path"].parent / sample["png"]
    next(upload for upload in app.file_uploader if upload.label == "晶圆文件").upload(png.name, png.read_bytes(), "image/png").run()
    next(button for button in app.button if button.label == "识别晶圆").click().run()
    assert not app.exception, app.exception
    assert next(metric for metric in app.metric if metric.label == "预测类别").value == expected["prediction"]
    next(upload for upload in app.file_uploader if upload.label == "晶圆文件").set_value(("invalid.png", b"invalid", "image/png")).run()
    next(button for button in app.button if button.label == "识别晶圆").click().run()
    assert any("输入读取失败" in error.value for error in app.error)
    print(f"PASS Streamlit: {task}, {path.stem}, {sample['id']} -> {expected['prediction']}; PNG upload component, rejection, batch and run log passed")


if __name__ == "__main__":
    main()
