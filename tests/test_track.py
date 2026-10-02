from twod23d.stages.track import keep_real_tracks


def test_keep_real_tracks():
    player = [{"frame": f, "track_id": 1, "score": 0.9} for f in range(30)]
    blip = [{"frame": f, "track_id": 2, "score": 0.9} for f in range(5)]
    unsure = [{"frame": f, "track_id": 3, "score": 0.3} for f in range(30)]
    kept = keep_real_tracks(player + blip + unsure, min_frames=30)
    assert kept == player
