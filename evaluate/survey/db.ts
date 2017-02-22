import {
    createConnection, Entity, Column,
    PrimaryColumn, PrimaryGeneratedColumn, Connection, OneToOne, OneToMany, ManyToOne,
    CreateDateColumn
} from 'typeorm';
import "reflect-metadata";

import { join } from 'path';

@Entity()
export class Session {
    @PrimaryGeneratedColumn()
    id: number;
    @Column({ nullable: true })
    bcSampleSource: string | null = null;
    @OneToMany(type => BCPrediction, bc => bc.session)
    bcs: BCPrediction[];
    @CreateDateColumn()
    created: Date;
    @Column('text')
    handshake: string;
    @OneToMany(type => NetRating, rating => rating.session)
    netRatings: NetRating[];
    @Column('text', { nullable: true })
    comment: string;
}
@Entity()
export class BCPrediction {
    @PrimaryGeneratedColumn()
    id: number;
    @ManyToOne(type => Session, session => session.bcs)
    session: Session;
    @Column()
    segment: string;
    @Column()
    time: number;
    @Column()
    duration: number;
    @CreateDateColumn()
    created: Date;
}
@Entity()
export class NetRating {
    @PrimaryGeneratedColumn()
    id: number;
    @ManyToOne(type => Session, session => session.netRatings)
    session: Session;
    @Column()
    segment: string;
    @Column("int", { nullable: true })
    rating: number | null;
    @Column()
    ratingType: string;
    @CreateDateColumn()
    created: Date;
    @Column()
    final: boolean;
}

export function openDatabase() {
    return createConnection({
        driver: {
            type: "sqlite",
            storage: join(__dirname, "db.sqlite")
        },
        entities: [
            Session, BCPrediction, NetRating
        ],
        autoSchemaSync: true
    })
}
