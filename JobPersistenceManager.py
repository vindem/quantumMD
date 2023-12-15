class JobPersistenceManager(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(JobPersistenceManager, cls).__new__(cls)
            cls.job_ids = set()
            return cls.instance
        return cls.instance

    def add_id(self, id):
        self.job_ids.add(id)
    def remove_id(self,id):
        self.job_ids.remove(id)

    def clear_all(self):
        self.job_ids.clear()

    def active_jobs(self):
        return self.job_ids