import pytest

torch = pytest.importorskip("torch")

from noisereduce.torchgate import TorchGate  # noqa: E402


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable")),
])
@pytest.mark.parametrize("nonstationary", [False, True])
@pytest.mark.parametrize("n_fft,hop_length,length", [
    (1024, 256, 4096),
    (1024, 256, 4107),
    (512, 100, 4123),
])
def test_torchgate_preserves_input_length(
    device, nonstationary, n_fft, hop_length, length
):
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(2, length, generator=generator, device=device)
    x.requires_grad_()
    gate = TorchGate(
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        nonstationary=nonstationary,
        prop_decrease=0,
        freq_mask_smooth_hz=None,
        time_mask_smooth_ms=None,
    ).to(device)

    result = gate(x)

    assert result.shape == x.shape
    torch.testing.assert_close(result, x)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
