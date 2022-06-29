import asyncio


async def moveMountConstantV(mount, startAngle, stopAngle):
    """Change the elevation angle step by step"""
    # change the elevation angle step by step

    freq = 1  # Hz
    vAngle = 2  # 1 deg change per minute
    holdMinutes = 0.1  # how long to hold at integeter values of the elevation angle
    angleStepSize = 1  # each time we change by 1 deg, before we hold in place

    rampMinutes = angleStepSize / vAngle
    print(
        "This will run for %.0f minutes"
        % ((startAngle - stopAngle) * (rampMinutes + holdMinutes))
    )
    start_time = Time(datetime.now())
    startTime = time.time()
    end_time = start_time + timedelta(minutes=80)
    demandAngle = startAngle
    while demandAngle > stopAngle - 0.01:
        await asyncio.sleep(1.0 / freq)

        timeNow = time.time()
        minutesEllapsed = (timeNow - startTime) / 60
        cyclePassed = np.floor(minutesEllapsed / (rampMinutes + holdMinutes))
        minutesIntoThisCycle = min(
            rampMinutes, minutesEllapsed - cyclePassed * (rampMinutes + holdMinutes)
        )
        demandAngle = (
            startAngle - cyclePassed * angleStepSize - minutesIntoThisCycle * vAngle
        )
        # print(demandAngle, cyclePassed, minutesIntoThisCycle)
        await mount.cmd_moveToTarget.set_start(azimuth=0, elevation=demandAngle)
