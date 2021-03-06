// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.config.server.maintenance;

import com.yahoo.log.LogLevel;
import com.yahoo.vespa.config.server.ApplicationRepository;
import com.yahoo.vespa.curator.Curator;
import com.yahoo.vespa.flags.FlagSource;
import com.yahoo.vespa.flags.Flags;
import com.yahoo.vespa.flags.LongFlag;

import java.time.Duration;

/**
 * Removes inactive sessions
 * <p>
 * Note: Unit test is in ApplicationRepositoryTest
 *
 * @author hmusum
 */
public class SessionsMaintainer extends ConfigServerMaintainer {
    private final boolean hostedVespa;
    private final LongFlag expiryTimeFlag;

    SessionsMaintainer(ApplicationRepository applicationRepository, Curator curator, Duration interval, FlagSource flagSource) {
        // Start this maintainer immediately. It frees disk space, so if disk goes full and config server
        // restarts this makes sure that cleanup will happen as early as possible
        super(applicationRepository, curator, Duration.ZERO, interval);
        this.hostedVespa = applicationRepository.configserverConfig().hostedVespa();
        this.expiryTimeFlag = Flags.CONFIGSERVER_SESSIONS_EXPIRY_INTERVAL_IN_DAYS.bindTo(flagSource);
    }

    @Override
    protected void maintain() {
        applicationRepository.deleteExpiredLocalSessions();

        // Expired remote sessions are sessions that belong to an application that have external deployments that
        // are no longer active
        if (hostedVespa) {
            Duration expiryTime = Duration.ofDays(expiryTimeFlag.value());
            int deleted = applicationRepository.deleteExpiredRemoteSessions(expiryTime);
            log.log(LogLevel.INFO, "Deleted " + deleted + " expired remote sessions, expiry time " + expiryTime);
        }
    }
}
