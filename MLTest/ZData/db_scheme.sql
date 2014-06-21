-- Tables for the Home_M.2010.ZigBee Dataset are in SQL compatible with PostgreSQL

-- Database: smart_home

-- DROP DATABASE smart_home;

CREATE DATABASE smart_home
  WITH OWNER = postgres
       ENCODING = 'UTF8'
       TABLESPACE = pg_default
       LC_COLLATE = 'en_CA.UTF-8'
       LC_CTYPE = 'en_CA.UTF-8'
       CONNECTION LIMIT = -1;

-- Table: daylight

-- DROP TABLE daylight;

CREATE TABLE daylight
(
  day_dt date NOT NULL,
  sunrise_tm time without time zone NOT NULL,
  sunset_tm time without time zone NOT NULL,
  noon_tm time without time zone NOT NULL,
  noon_altitude numeric(4,1) NOT NULL,
  noon_distance numeric(7,3) NOT NULL,
  CONSTRAINT daylight_pk PRIMARY KEY (day_dt )
)
WITH (
  OIDS=FALSE
);
ALTER TABLE daylight
  OWNER TO postgres;
GRANT ALL ON TABLE daylight TO postgres;
GRANT SELECT ON TABLE daylight TO smuser;

-- Table: weather

-- DROP TABLE weather;

CREATE TABLE weather
(
  read_dt date NOT NULL,
  read_tm time without time zone NOT NULL,
  condition character varying(32),
  temperature numeric(4,1),
  pressure numeric(4,1),
  humidity numeric(4,1),
  wind_direction character varying(3),
  wind_speed numeric(4,1),
  air_quality_health_index numeric(4,1),
  CONSTRAINT weather_pk PRIMARY KEY (read_dt , read_tm )
)
WITH (
  OIDS=FALSE
);
ALTER TABLE weather
  OWNER TO postgres;
GRANT ALL ON TABLE weather TO postgres;
GRANT SELECT, UPDATE, INSERT ON TABLE weather TO smuser;

-- Table: environment

-- DROP TABLE environment;

CREATE TABLE environment
(
  read_dt date NOT NULL,
  read_tm time without time zone NOT NULL,
  equipment_id character varying(32) NOT NULL,
  temperature numeric(3,1),
  light numeric(5,1),
  CONSTRAINT environment_pk PRIMARY KEY (read_dt , read_tm , equipment_id ),
  CONSTRAINT environment_equipment_id FOREIGN KEY (equipment_id)
      REFERENCES equipment (equipment_id) MATCH SIMPLE
      ON UPDATE RESTRICT ON DELETE RESTRICT
)
WITH (
  OIDS=FALSE
);
ALTER TABLE environment
  OWNER TO postgres;
GRANT ALL ON TABLE environment TO postgres;
GRANT SELECT, UPDATE, INSERT ON TABLE environment TO smuser;

-- Table: equipment

-- DROP TABLE equipment;

CREATE TABLE equipment
(
  equipment_id character varying(32) NOT NULL,
  manufacturer character varying(64) NOT NULL,
  model character varying(64) NOT NULL,
  product_type character varying(128) NOT NULL,
  node_type character varying(32) NOT NULL,
  network_type character varying(32) NOT NULL,
  house_zone character varying(32) NOT NULL,
  readings_table character varying(32),
  display_name character varying(32),
  CONSTRAINT equipment_pk PRIMARY KEY (equipment_id )
)
WITH (
  OIDS=FALSE
);
ALTER TABLE equipment
  OWNER TO postgres;
GRANT ALL ON TABLE equipment TO postgres;
GRANT SELECT ON TABLE equipment TO smuser;

-- Table: power_meter

-- DROP TABLE power_meter;

CREATE TABLE power_meter
(
  read_dt date NOT NULL,
  read_tm time without time zone NOT NULL,
  equipment_id character varying(32) NOT NULL,
  kwh_del integer,
  kwh_rec integer,
  kwh_total integer,
  kwh_interval integer,
  kvarh_del integer,
  kvarh_rec integer,
  kvarh_total integer,
  kvarh_interval integer,
  kvah_total integer,
  kvah_interval integer,
  v_avg real,
  i_avg real,
  freq real,
  pf_sign real,
  kw_total real,
  kvar_total real,
  kva_total real,
  CONSTRAINT power_meter_pk PRIMARY KEY (read_dt , read_tm , equipment_id ),
  CONSTRAINT power_meter_equipment_id FOREIGN KEY (equipment_id)
      REFERENCES equipment (equipment_id) MATCH SIMPLE
      ON UPDATE RESTRICT ON DELETE RESTRICT
)
WITH (
  OIDS=FALSE
);
ALTER TABLE power_meter
  OWNER TO postgres;
GRANT ALL ON TABLE power_meter TO postgres;
GRANT SELECT, UPDATE, INSERT ON TABLE power_meter TO smuser;

